//! Splits model output into reasoning, content and tool calls (Qwen XML tool format).

use serde_json::{Value, json};

pub const THINK_END: &str = "</think>";
pub const TOOL_START: &str = "<tool_call>";

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
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
    fn holdback_partial_tags() {
        assert_eq!(holdback("abc</thi", &[THINK_END]), 5);
        assert_eq!(holdback("abc", &[THINK_END]), 0);
        assert_eq!(holdback("x<", &[THINK_END, TOOL_START]), 1);
    }
}
