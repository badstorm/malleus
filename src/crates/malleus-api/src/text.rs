//! Splits model output into reasoning, content and tool calls: the `thinking`/`<tool_call>` XML of the Qwen family,
//! and the message tokens of the families whose template writes them (Inkling-2).

use serde_json::{Value, json};

pub const THINK_END: &str = "</think>";
pub const TOOL_START: &str = "<tool_call>";

/// The message-token format: an assistant turn is one or more segments, each opened by a role token and a content
/// token and closed by `<|end_message|>`. Thinking, text and a tool call are three segments, so the same turn can
/// carry both a reasoning block and a call.
pub const MSG_MODEL: &str = "<|message_model|>";
pub const MSG_THINK: &str = "<|content_thinking|>";
pub const MSG_TEXT: &str = "<|content_text|>";
pub const MSG_TOOL: &str = "<|content_invoke_tool_json|>";
pub const MSG_END: &str = "<|end_message|>";
pub const MSG_EOS: &str = "<|content_model_end_sampling|>";

const MSG_MARKERS: &[&str] = &[MSG_MODEL, MSG_THINK, MSG_TEXT, MSG_TOOL, MSG_END, MSG_EOS];

/// The segment a message token opened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Seg {
    None,
    /// After `<|message_model|>`: the segment's name, which is a tool's name when `<|content_invoke_tool_json|>`
    /// comes next and empty otherwise.
    Name,
    Think,
    Text,
    Tool,
    /// The turn ended (`<|content_model_end_sampling|>`): nothing more is read.
    Done,
}

fn segment(seg: Seg) -> Option<Seg> {
    match seg {
        Seg::Name => None,
        s => Some(s),
    }
}

/// The first message marker in `s`, with its position.
fn next_marker(s: &str) -> Option<(usize, &'static str)> {
    MSG_MARKERS
        .iter()
        .filter_map(|m| s.find(m).map(|i| (i, *m)))
        .min_by_key(|(i, _)| *i)
}

/// `{"name": ..., "args": {...}}` as a tool call. Anything else is not a call and is kept as text.
fn push_tool(p: &mut Parsed, name: &str, json_text: &str) -> bool {
    let Ok(v) = serde_json::from_str::<Value>(json_text) else {
        return false;
    };
    let Some(found) = v["name"].as_str().or_else(|| Some(name).filter(|n| !n.is_empty())) else {
        return false;
    };
    // The template writes the arguments twice (a `name` field and `args`); `args` is the one an OpenAI client reads.
    let arguments = if v.get("args").is_some() {
        v["args"].clone()
    } else {
        v.clone()
    };
    p.tool_calls.push(ToolCall {
        name: found.to_string(),
        arguments,
    });
    true
}

/// The output of a family whose template writes message tokens (Inkling-2): every segment of the assistant turn
/// becomes reasoning, text or a tool call.
pub fn parse_message_output(out: &str, tools: bool) -> Parsed {
    let mut p = Parsed::default();
    let mut seg = Seg::None;
    let mut name = String::new();
    let mut call = String::new();
    let mut rest = out;
    while let Some((i, m)) = next_marker(rest) {
        let text = &rest[..i];
        match seg {
            Seg::Think => p.reasoning.push_str(text),
            Seg::Text => p.content.push_str(text),
            Seg::Name => name.push_str(text),
            Seg::Tool => call.push_str(text),
            Seg::None | Seg::Done => {}
        }
        rest = &rest[i + m.len()..];
        match m {
            MSG_MODEL => seg = Seg::Name,
            MSG_THINK => seg = Seg::Think,
            MSG_TEXT => seg = Seg::Text,
            MSG_TOOL => seg = Seg::Tool,
            MSG_END => {
                // The arguments are the only segment whose content is not text: a call the client did not declare is
                // kept as text rather than dropped.
                if seg == Seg::Tool && !(tools && push_tool(&mut p, &name, &call)) {
                    p.content.push_str(&name);
                    p.content.push_str(&call);
                }
                seg = Seg::None;
                name.clear();
                call.clear();
            }
            _ => {
                seg = Seg::Done;
                break;
            }
        }
    }
    match seg {
        Seg::Think => p.reasoning.push_str(rest),
        Seg::Text => p.content.push_str(rest),
        Seg::Name => name.push_str(rest),
        Seg::Tool => {
            if !(tools && push_tool(&mut p, &name, rest)) {
                p.content.push_str(&name);
                p.content.push_str(rest);
            }
        }
        Seg::None | Seg::Done => {}
    }
    p.reasoning = p.reasoning.trim().to_string();
    p.content = p.content.trim().to_string();
    p
}

/// [`parse_message_output`], incrementally: the segments that are text become deltas as they arrive, and the tool
/// calls are read off the whole output once the turn ends (`parse_message_output`), so they are not emitted here.
pub struct MessageSplitter {
    seg: Seg,
    buf: String,
}

impl Default for MessageSplitter {
    fn default() -> Self {
        Self::new()
    }
}

impl MessageSplitter {
    pub fn new() -> Self {
        MessageSplitter {
            seg: Seg::None,
            buf: String::new(),
        }
    }

    fn emit(&self, out: &mut Vec<Delta>, text: &str) {
        if text.is_empty() {
            return;
        }
        match segment(self.seg) {
            Some(Seg::Think) => out.push(Delta::Reasoning(text.to_string())),
            Some(Seg::Text) => out.push(Delta::Content(text.to_string())),
            _ => {}
        }
    }

    pub fn push(&mut self, s: &str, last: bool) -> Vec<Delta> {
        self.buf.push_str(s);
        let mut out = Vec::new();
        loop {
            if self.seg == Seg::Done {
                self.buf.clear();
                return out;
            }
            let Some((i, m)) = next_marker(&self.buf) else {
                let keep = if last { 0 } else { holdback(&self.buf, MSG_MARKERS) };
                let n = self.buf.len() - keep;
                if n > 0 {
                    let text = self.buf[..n].to_string();
                    self.buf.drain(..n);
                    self.emit(&mut out, &text);
                }
                return out;
            };
            let text = self.buf[..i].to_string();
            self.buf.drain(..i + m.len());
            self.emit(&mut out, &text);
            self.seg = match m {
                MSG_MODEL => Seg::Name,
                MSG_THINK => Seg::Think,
                MSG_TEXT => Seg::Text,
                MSG_TOOL => Seg::Tool,
                MSG_END => Seg::None,
                _ => Seg::Done,
            };
            // The name of a tool and its JSON are dropped here: they are read from the whole output at the end.
            if self.seg == Seg::Name || self.seg == Seg::Tool {
                self.buf.clear();
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

/// One piece of a streamed answer: reasoning goes to `reasoning_content`, text to `content`.
#[derive(Debug)]
pub enum Delta {
    Reasoning(String),
    Content(String),
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Parsed {
    pub reasoning: String,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

fn parse_value(v: &str) -> Value {
    serde_json::from_str(v).unwrap_or_else(|_| Value::String(v.to_string()))
}

pub fn parse_tool_calls(s: &str) -> (String, Vec<ToolCall>) {
    let mut calls = Vec::new();
    let mut rest = s;
    let mut text = String::new();
    while let Some(i) = rest.find(TOOL_START) {
        text.push_str(&rest[..i]);
        let after = &rest[i + TOOL_START.len()..];
        let Some(end) = after.find("</tool_call>") else {
            text.push_str(&rest[i..]);
            rest = "";
            break;
        };
        let body = &after[..end];
        if let Some(f) = body.find("<function=") {
            let fb = &body[f + "<function=".len()..];
            if let Some(gt) = fb.find('>') {
                let name = fb[..gt].trim().to_string();
                let mut args = serde_json::Map::new();
                let mut p = &fb[gt + 1..];
                while let Some(ps) = p.find("<parameter=") {
                    let pb = &p[ps + "<parameter=".len()..];
                    let Some(pgt) = pb.find('>') else { break };
                    let pname = pb[..pgt].trim().to_string();
                    let val_start = &pb[pgt + 1..];
                    let Some(pend) = val_start.find("</parameter>") else {
                        break;
                    };
                    let raw = val_start[..pend].strip_prefix('\n').unwrap_or(&val_start[..pend]);
                    let raw = raw.strip_suffix('\n').unwrap_or(raw);
                    args.insert(pname, parse_value(raw));
                    p = &val_start[pend + "</parameter>".len()..];
                }
                calls.push(ToolCall {
                    name,
                    arguments: Value::Object(args),
                });
            }
        }
        rest = &after[end + "</tool_call>".len()..];
    }
    text.push_str(rest);
    (text.trim().to_string(), calls)
}

/// `thinking`: the prompt ended inside an open `<think>` block, so the output starts with reasoning.
pub fn parse_output(out: &str, thinking: bool, tools: bool) -> Parsed {
    let (reasoning, content) = if thinking {
        match out.find(THINK_END) {
            Some(i) => (out[..i].trim().to_string(), out[i + THINK_END.len()..].to_string()),
            None => (out.trim().to_string(), String::new()),
        }
    } else {
        (String::new(), out.to_string())
    };
    let (content, tool_calls) = if tools {
        parse_tool_calls(&content)
    } else {
        (content.trim().to_string(), vec![])
    };
    Parsed {
        reasoning,
        content,
        tool_calls,
    }
}

/// Length of the longest suffix of `s` that is a proper prefix of one of `tags` (text that must be held back).
pub fn holdback(s: &str, tags: &[&str]) -> usize {
    let mut best = 0;
    for t in tags {
        for n in (1..t.len()).rev() {
            if n <= s.len() && s.is_char_boundary(s.len() - n) && t.starts_with(&s[s.len() - n..]) {
                best = best.max(n);
                break;
            }
        }
    }
    best
}

pub fn openai_tool_calls(calls: &[ToolCall], id_base: &str) -> Value {
    Value::Array(
        calls
            .iter()
            .enumerate()
            .map(|(i, c)| {
                json!({
                    "id": format!("call_{id_base}_{i}"),
                    "type": "function",
                    "function": { "name": c.name, "arguments": c.arguments.to_string() }
                })
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_reasoning_and_tools() {
        let out = "let me think\n</think>\n\nI'll check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nRoma\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
        let p = parse_output(out, true, true);
        assert_eq!(p.reasoning, "let me think");
        assert_eq!(p.content, "I'll check.");
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.tool_calls[0].name, "get_weather");
        assert_eq!(p.tool_calls[0].arguments, json!({"city": "Roma", "days": 3}));
    }

    #[test]
    fn message_tokens_split_into_reasoning_text_and_calls() {
        let out = "<|message_model|><|content_thinking|>need a tool<|end_message|>\
<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Roma\"}}<|end_message|>\
<|message_model|><|content_text|>Ecco il meteo.<|end_message|><|content_model_end_sampling|>";
        let p = parse_message_output(out, true);
        assert_eq!(p.reasoning, "need a tool");
        assert_eq!(p.content, "Ecco il meteo.");
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.tool_calls[0].name, "get_weather");
        assert_eq!(p.tool_calls[0].arguments, json!({ "city": "Roma" }));
        // Without tools declared the call is kept as text instead of being dropped.
        let p = parse_message_output(out, false);
        assert!(p.tool_calls.is_empty());
        assert!(p.content.contains("get_weather"));
    }

    #[test]
    fn message_splitter_streams_text_and_drops_the_call() {
        let out = "<|message_model|><|content_thinking|>think<|end_message|>\
<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{}}<|end_message|>\
<|message_model|><|content_text|>Ciao!<|end_message|>";
        let mut sp = MessageSplitter::new();
        let mut reasoning = String::new();
        let mut content = String::new();
        // Byte by byte, the worst case for the holdback.
        let n = out.len();
        for (i, c) in out.char_indices() {
            let last = i + c.len_utf8() == n;
            for d in sp.push(&out[i..i + c.len_utf8()], last) {
                match d {
                    Delta::Reasoning(r) => reasoning.push_str(&r),
                    Delta::Content(c) => content.push_str(&c),
                }
            }
        }
        assert_eq!(reasoning, "think");
        assert_eq!(content, "Ciao!");
    }

    #[test]
    fn holdback_partial_tags() {
        assert_eq!(holdback("abc</thi", &[THINK_END]), 5);
        assert_eq!(holdback("abc", &[THINK_END]), 0);
        assert_eq!(holdback("x<", &[THINK_END, TOOL_START]), 1);
    }
}
