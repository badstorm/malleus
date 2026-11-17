#![forbid(unsafe_code)]

use anyhow::{Context, Result, anyhow};
use minijinja::{Environment, Error, ErrorKind, Value};
use serde_json::json;
use std::path::Path;
use tokenizers::Tokenizer;

pub struct Chat {
    tok: Tokenizer,
    env: Environment<'static>,
    template: String,
}

pub struct RenderOptions<'a> {
    pub add_generation_prompt: bool,
    pub enable_thinking: Option<bool>,
    pub reasoning_effort: Option<&'a str>,
    pub tools: Option<&'a serde_json::Value>,
}

impl Default for RenderOptions<'_> {
    fn default() -> Self {
        RenderOptions {
            add_generation_prompt: true,
            enable_thinking: None,
            reasoning_effort: None,
            tools: None,
        }
    }
}

fn raise_exception(msg: String) -> Result<Value, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, msg))
}

impl Chat {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let tok = std::fs::read(model_dir.join("tokenizer.json")).context("tokenizer.json")?;
        let tpl = std::fs::read_to_string(model_dir.join("chat_template.jinja")).context("chat_template.jinja")?;
        Self::from_parts(&tok, tpl)
    }

    pub fn from_parts(tokenizer_json: &[u8], template: String) -> Result<Self> {
        let tok = Tokenizer::from_bytes(tokenizer_json).map_err(|e| anyhow!("tokenizer: {e}"))?;
        let tpl = template.clone();
        let mut env = Environment::new();
        minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", raise_exception);
        env.set_trim_blocks(false);
        env.add_template_owned("chat", tpl)?;
        Ok(Chat { tok, env, template })
    }

    /// The Jinja template as it was loaded. The API serves it to clients that decide what a model supports from it
    /// (a template with `enable_thinking` takes the reasoning flag, an `image_pad` tokenizer takes images).
    pub fn template(&self) -> &str {
        &self.template
    }

    pub fn render(&self, messages: &serde_json::Value, o: &RenderOptions) -> Result<String> {
        let mut ctx = json!({ "messages": messages, "add_generation_prompt": o.add_generation_prompt });
        if let Some(t) = o.enable_thinking {
            ctx["enable_thinking"] = json!(t);
        }
        if let Some(r) = o.reasoning_effort {
            ctx["reasoning_effort"] = json!(r);
        }
        if let Some(t) = o.tools {
            ctx["tools"] = t.clone();
        }
        Ok(self.env.get_template("chat")?.render(Value::from_serialize(&ctx))?)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tok
            .encode(text, false)
            .map_err(|e| anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tok.decode(ids, false).map_err(|e| anyhow!("decode: {e}"))
    }

    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.tok.token_to_id(s)
    }

    /// The bytes a token id stands for. Byte-level BPE spells each raw byte as one character of the GPT-2 alphabet,
    /// so the raw piece is mapped back through that alphabet; a piece that uses a character outside it (an added
    /// token, or a non-byte-level vocabulary) is taken literally, as UTF-8. This is the same reconstruction
    /// `bench/quality_api.py` builds from the tokenizer, so the engine and the reference tool compare the same bytes.
    pub fn token_bytes(&self, id: u32) -> Option<Vec<u8>> {
        let piece = self.tok.id_to_token(id)?;
        Some(piece_to_bytes(&piece))
    }

    /// Ids of the tokens whose text carries at least one ASCII or Latin-1/Extended character. The draft head can
    /// then be a subset of `lm_head` (MALLEUS_DRAFT_VOCAB_FILE): punctuation, digits and Latin letters stay,
    /// pure CJK/Cyrillic/Arabic tokens leave. Sorted by id, as the loader expects.
    pub fn latin_vocab_ids(&self) -> Vec<u32> {
        let n = self.tok.get_vocab_size(true);
        let mut out = Vec::with_capacity(n);
        for id in 0..n as u32 {
            // Decode rather than read the raw token: byte-level BPE stores CJK as Latin-1 byte fragments, and
            // those would all pass a check on the raw string.
            if let Ok(s) = self.tok.decode(&[id], false)
                && s.chars()
                    .any(|c| c.is_ascii() || ('\u{00A0}'..='\u{024F}').contains(&c))
            {
                out.push(id);
            }
        }
        out
    }
}

/// The byte a GPT-2 byte-level character stands for — the inverse of the alphabet `bench/quality_api.py` builds
/// ("!".."~", "\xA1".."\xAC", "\xAE".."\xFF", and the remaining control bytes after `U+0100`).
fn byte_of(c: char) -> Option<u8> {
    let u = c as u32;
    match u {
        0x21..=0x7E => Some(u as u8),
        0xA1..=0xAC => Some(u as u8),
        0xAE..=0xFF => Some(u as u8),
        0x100..=0x120 => Some((u - 0x100) as u8),
        0x121 => Some(0x7F),
        0x122..=0x142 => Some((u - 0x122) as u8 + 0x80),
        0x143 => Some(0xAD),
        _ => None,
    }
}

/// Rebuilds the bytes of a raw BPE piece: every character goes through the byte-level alphabet, and the whole piece
/// is taken literally (its UTF-8 bytes) as soon as one of them does not — added tokens and non-byte-level vocabularies.
fn piece_to_bytes(piece: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(piece.len());
    for c in piece.chars() {
        match byte_of(c) {
            Some(b) => out.push(b),
            None => return piece.as_bytes().to_vec(),
        }
    }
    out
}

/// Incremental detokenizer that only emits text once it is valid UTF-8.
pub struct StreamDecoder {
    ids: Vec<u32>,
    emitted: usize,
}

impl Default for StreamDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamDecoder {
    pub fn new() -> Self {
        StreamDecoder {
            ids: Vec::new(),
            emitted: 0,
        }
    }

    pub fn push(&mut self, chat: &Chat, id: u32) -> Result<String> {
        self.ids.push(id);
        let text = chat.decode(&self.ids)?;
        if text.ends_with('\u{FFFD}') || text.len() < self.emitted {
            return Ok(String::new());
        }
        let out = text[self.emitted..].to_string();
        self.emitted = text.len();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The GPT-2 byte alphabet spells bytes missing from the printable ranges as `U+0100 + index` in the ascending
    /// missing list (0x00..0x20, 0x7F, 0x80..0xA0, 0xAD). Every byte must round-trip.
    fn missing_index(b: u8) -> u8 {
        match b {
            0x00..=0x20 => b,
            0x7F => 0x21,
            0x80..=0xA0 => b - 0x5E,
            0xAD => 0x43,
            _ => panic!("byte {b:#04x} is printable"),
        }
    }

    #[test]
    fn byte_alphabet_round_trips_every_byte() {
        for b in 0u16..=255 {
            let c = match b {
                0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF => char::from_u32(b as u32).unwrap(),
                _ => char::from_u32(0x100 + missing_index(b as u8) as u32).unwrap(),
            };
            assert_eq!(byte_of(c), Some(b as u8), "byte {b:#04x}");
        }
    }

    #[test]
    fn pieces_are_rebuilt_as_bytes() {
        // "Ġ" is the space byte, "Ċ" the newline byte, "é" (Latin-1) the raw 0xE9 byte;
        // a piece with a character outside the byte alphabet is taken literally.
        assert_eq!(piece_to_bytes("ĠHello"), b" Hello".to_vec());
        assert_eq!(piece_to_bytes("Ċ"), b"\n".to_vec());
        assert_eq!(piece_to_bytes("<|im_start|>"), b"<|im_start|>".to_vec());
        assert_eq!(piece_to_bytes("é"), vec![0xE9]);
        assert_eq!(piece_to_bytes("♥a"), "♥a".as_bytes());
    }
}
