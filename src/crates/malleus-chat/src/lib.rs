#![forbid(unsafe_code)]

use anyhow::{Context, Result, anyhow};
use minijinja::{Environment, Error, ErrorKind, Value};
use serde_json::json;
use std::path::Path;
use tokenizers::Tokenizer;

pub struct Chat {
    tok: Tokenizer,
    env: Environment<'static>,
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
        let tpl = template;
        let mut env = Environment::new();
        minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", raise_exception);
        env.set_trim_blocks(false);
        env.add_template_owned("chat", tpl)?;
        Ok(Chat { tok, env })
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
