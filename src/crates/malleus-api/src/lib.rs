#![forbid(unsafe_code)]

pub mod text;

use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine as _;
use futures_util::stream::Stream;
use malleus_chat::{Chat, RenderOptions, StreamDecoder};
use malleus_sched::{Engine, Event, Finish, FinishReason, Request, Sampling};
use serde_json::{Value, json};
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use text::{THINK_END, TOOL_START, holdback, openai_tool_calls, parse_output};
use tokio::sync::mpsc;

pub struct Config {
    pub model_name: String,
    pub api_key: Option<String>,
    pub default_thinking: bool,
    pub max_tokens_default: usize,
}

pub struct AppState {
    pub engine: Engine,
    pub chat: Chat,
    pub cfg: Config,
    pub eos: Vec<u32>,
    counter: AtomicU64,
}

type Shared = Arc<AppState>;

pub fn router(engine: Engine, chat: Chat, cfg: Config, eos: Vec<u32>) -> Router {
    let st = Arc::new(AppState {
        engine,
        chat,
        cfg,
        eos,
        counter: AtomicU64::new(0),
    });
    Router::new()
        .route("/", get(landing))
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/messages/count_tokens", post(anthropic_count_tokens))
        .layer(DefaultBodyLimit::max(64 << 20))
        .with_state(st)
}

#[derive(Debug)]
struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = json!({ "error": { "message": self.1, "type": "invalid_request_error" } });
        (self.0, Json(body)).into_response()
    }
}

fn bad(msg: impl Into<String>) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, msg.into())
}

fn check_auth(st: &AppState, h: &HeaderMap) -> Result<(), ApiError> {
    let Some(key) = &st.cfg.api_key else { return Ok(()) };
    let bearer = h
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    let xkey = h.get("x-api-key").and_then(|v| v.to_str().ok());
    if bearer == Some(key.as_str()) || xkey == Some(key.as_str()) {
        Ok(())
    } else {
        Err(ApiError(StatusCode::UNAUTHORIZED, "invalid API key".into()))
    }
}

/// What an unauthenticated visitor gets when the server has an API key: the wordmark, nothing else. No model name,
/// no endpoint list, no hint that a key is configured.
const LOGO: &str = r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Malleus</title>
<style>
html,body{height:100%;margin:0}
body{display:flex;align-items:center;justify-content:center;background:#fff;color:#1b1b1b;
font-family:system-ui,-apple-system,Segoe UI,sans-serif}
h1{font-size:2.6rem;font-weight:600;letter-spacing:-.03em;margin:0}
</style></head>
<body><h1>Malleus</h1></body></html>
"#;

/// Landing page on `/`: one self-contained HTML page, no external assets, so opening the container's address in a
/// browser says what is running instead of answering 404. The API itself stays under `/v1`.
///
/// The endpoints and the model name are for an authenticated client only: the full page is what a request that
/// carries the configured key gets, and the wordmark is everything else. That includes the case where the server has
/// no key at all — there is no session that could present one, so the page stays the wordmark. Browsers do not send an
/// `Authorization` header on a plain visit, so an operator sees the logo; clients send `Bearer` and get the list.
async fn landing(State(st): State<Shared>, headers: HeaderMap) -> Html<String> {
    let authenticated = st.cfg.api_key.is_some() && check_auth(&st, &headers).is_ok();
    if !authenticated {
        return Html(LOGO.to_string());
    }
    fn esc(s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
    }
    let model = esc(&st.cfg.model_name);
    Html(format!(
        r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Malleus</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:42rem;margin:4rem auto;padding:0 1.2rem;
line-height:1.6;color:#1b1b1b;background:#fff}}
h1{{font-size:2.1rem;margin:0 0 .2rem;letter-spacing:-.02em}}
p.sub{{color:#666;margin:0 0 1.6rem}}
code{{background:#f2f2f2;padding:.1rem .35rem;border-radius:4px;font-size:.92em}}
li{{margin:.25rem 0}} ul{{padding-left:1.2rem}}
footer{{margin-top:2.5rem;color:#888;font-size:.85rem;border-top:1px solid #eee;padding-top:1rem}}
</style></head>
<body>
<h1>Malleus</h1>
<p class="sub">Open-source LLM inference engine for AMD RDNA 3.5 APUs (gfx1150 and gfx1151).</p>
<p>Serving <code>{model}</code> on this port. The API is OpenAI-compatible and lives under <code>/v1</code>:</p>
<ul>
<li><code>GET /health</code> &mdash; liveness and queue depth</li>
<li><code>GET /v1/models</code> &mdash; the model served here</li>
<li><code>POST /v1/chat/completions</code> &mdash; chat, OpenAI schema (streaming, tools)</li>
<li><code>POST /v1/messages</code> &mdash; chat, Anthropic schema</li>
<li><code>POST /v1/messages/count_tokens</code> &mdash; token counting</li>
</ul>
<footer>Runs entirely on the GPU the container was given. No outbound network connections.</footer>
</body></html>
"#
    ))
}

async fn health(State(st): State<Shared>) -> Json<Value> {
    Json(json!({ "status": "ok", "queued": st.engine.queued.load(Ordering::Relaxed) }))
}

async fn models(State(st): State<Shared>, h: HeaderMap) -> Result<Json<Value>, ApiError> {
    check_auth(&st, &h)?;
    Ok(Json(json!({
        "object": "list",
        "data": [{ "id": st.cfg.model_name, "object": "model", "owned_by": "malleus", "context_length": st.engine.ctx }]
    })))
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn content_text(v: &Value, what: &str) -> Result<String, ApiError> {
    match v {
        Value::Null => Ok(String::new()),
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                match p["type"].as_str() {
                    Some("text") | Some("input_text") | Some("output_text") => {
                        out.push_str(p["text"].as_str().unwrap_or(""))
                    }
                    Some("image_url") | Some("image") | Some("input_image") => {
                        return Err(bad("image input is not supported in this message"));
                    }
                    _ => return Err(bad(format!("unsupported content part in {what}"))),
                }
            }
            Ok(out)
        }
        _ => Err(bad(format!("invalid content in {what}"))),
    }
}

/// Raw bytes of an image part: a `data:` URL (OpenAI) or the base64 `source` of the Anthropic API.
fn image_bytes(p: &Value) -> Result<Vec<u8>, ApiError> {
    let url = p["image_url"]["url"]
        .as_str()
        .or(p["image_url"].as_str())
        .or(p["image"].as_str())
        .or(p["source"]["data"].as_str())
        .ok_or_else(|| bad("image part without a data: URL or a base64 source"))?;
    if url.starts_with("http://") || url.starts_with("https://") {
        return Err(bad(
            "image URLs are not fetched (no outbound connections); send a data: URL",
        ));
    }
    let b64 = url
        .strip_prefix("data:")
        .and_then(|s| s.split_once(','))
        .map_or(url, |(_, b)| b);
    base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|e| bad(format!("invalid base64 image: {e}")))
}

/// Content as the chat template expects it, plus the images it carries in order of appearance.
/// Text-only content is flattened to a string (the historic path); an array with images is kept as
/// an array so the Qwen template can emit `<|vision_start|><|image_pad|><|vision_end|>`.
fn content_parts(v: &Value, what: &str) -> Result<(Value, Vec<Vec<u8>>), ApiError> {
    match v {
        Value::Null | Value::String(_) => Ok((json!(content_text(v, what)?), vec![])),
        Value::Array(parts) => {
            let mut images = Vec::new();
            for p in parts {
                match p["type"].as_str() {
                    Some("text") | Some("input_text") | Some("output_text") => {}
                    Some("image_url") | Some("image") | Some("input_image") => {
                        if what != "user" {
                            return Err(bad(format!("images are not supported in {what} messages")));
                        }
                        images.push(image_bytes(p)?);
                    }
                    _ => return Err(bad(format!("unsupported content part in {what}"))),
                }
            }
            if images.is_empty() {
                Ok((json!(content_text(v, what)?), vec![]))
            } else {
                Ok((v.clone(), images))
            }
        }
        _ => Err(bad(format!("invalid content in {what}"))),
    }
}

/// OpenAI messages -> chat template messages, plus the raw images the template will place.
fn openai_messages(msgs: &Value) -> Result<(Value, Vec<Vec<u8>>), ApiError> {
    let arr = msgs.as_array().ok_or_else(|| bad("messages must be an array"))?;
    let mut out = Vec::new();
    let mut images = Vec::new();
    for m in arr {
        let role = match m["role"].as_str().unwrap_or("") {
            "developer" => "system",
            r @ ("system" | "user" | "assistant" | "tool") => r,
            r => return Err(bad(format!("unsupported role {r:?}"))),
        };
        let (content, imgs) = content_parts(&m["content"], role)?;
        images.extend(imgs);
        let mut o = json!({ "role": role, "content": content });
        if let Some(r) = m["reasoning_content"].as_str() {
            o["reasoning_content"] = json!(r);
        }
        if let Some(tc) = m["tool_calls"].as_array() {
            let calls: Vec<Value> = tc
                .iter()
                .map(|c| {
                    let args = &c["function"]["arguments"];
                    let args = match args {
                        Value::String(s) => serde_json::from_str(s).unwrap_or(json!({})),
                        v => v.clone(),
                    };
                    json!({ "function": { "name": c["function"]["name"], "arguments": args } })
                })
                .collect();
            o["tool_calls"] = json!(calls);
        }
        out.push(o);
    }
    Ok((Value::Array(out), images))
}

fn anthropic_messages_to_template(body: &Value) -> Result<(Value, Vec<Vec<u8>>), ApiError> {
    let mut out = Vec::new();
    let mut images = Vec::new();
    match &body["system"] {
        Value::Null => {}
        s => out.push(json!({ "role": "system", "content": content_text(s, "system")? })),
    }
    for m in body["messages"]
        .as_array()
        .ok_or_else(|| bad("messages must be an array"))?
    {
        let role = m["role"].as_str().unwrap_or("");
        let blocks = match &m["content"] {
            Value::String(s) => vec![json!({ "type": "text", "text": s })],
            Value::Array(a) => a.clone(),
            _ => return Err(bad("invalid message content")),
        };
        let mut text = String::new();
        let mut reasoning = String::new();
        let mut calls = Vec::new();
        let mut parts = Vec::new();
        let mut has_image = false;
        for b in &blocks {
            match b["type"].as_str().unwrap_or("") {
                "text" => {
                    let t = b["text"].as_str().unwrap_or("");
                    text.push_str(t);
                    parts.push(json!({ "type": "text", "text": t }));
                }
                "thinking" => reasoning.push_str(b["thinking"].as_str().unwrap_or("")),
                "redacted_thinking" => {}
                "tool_use" => calls.push(json!({ "function": { "name": b["name"], "arguments": b["input"] } })),
                "tool_result" => {
                    if has_image {
                        return Err(bad("images mixed with tool results are not supported"));
                    }
                    if !text.is_empty() {
                        out.push(json!({ "role": role, "content": std::mem::take(&mut text) }));
                        parts.clear();
                    }
                    let c = match &b["content"] {
                        Value::Array(a) => {
                            let only_text: Vec<Value> = a.iter().filter(|x| x["type"] == "text").cloned().collect();
                            content_text(&Value::Array(only_text), "tool_result")?
                        }
                        v => content_text(v, "tool_result")?,
                    };
                    out.push(json!({ "role": "tool", "content": c }));
                }
                "image" => {
                    if role != "user" {
                        return Err(bad("images are only supported in user messages"));
                    }
                    has_image = true;
                    images.push(image_bytes(b)?);
                    parts.push(b.clone());
                }
                t => return Err(bad(format!("unsupported content block {t:?}"))),
            }
        }
        if role == "assistant" {
            if has_image {
                return Err(bad("images are only supported in user messages"));
            }
            let mut o = json!({ "role": "assistant", "content": text });
            if !reasoning.is_empty() {
                o["reasoning_content"] = json!(reasoning);
            }
            if !calls.is_empty() {
                o["tool_calls"] = json!(calls);
            }
            out.push(o);
        } else if has_image {
            out.push(json!({ "role": role, "content": Value::Array(parts) }));
        } else if !text.is_empty() {
            out.push(json!({ "role": role, "content": text }));
        }
    }
    Ok((Value::Array(out), images))
}

struct Prepared {
    ids: Vec<u32>,
    /// Token sequences prompt lookup may draft from besides the prompt itself: the original text of the recent
    /// messages, with the line numbers and the JSON escapes a tool result arrives with removed (see `original_text`).
    drafts: Vec<Vec<u32>>,
    images: Vec<Vec<u8>>,
    image_token: Option<u32>,
    thinking: bool,
    tools: bool,
    max_tokens: usize,
    sampling: Sampling,
    stops: Vec<String>,
    /// Per-token log-probabilities in the response (OpenAI `logprobs` / completions `logprobs: int`);
    /// false keeps the engine path byte-identical to today.
    logprobs: bool,
    /// Number of top alternatives per token, capped at 20 like OpenAI; 0 = none.
    top_logprobs: usize,
}

/// Token sequences of the message texts as the model will write them back. An agent reads a file through a
/// line-numbered or JSON-escaped rendering and then rewrites the file without either, so a prompt lookup that only
/// sees the prompt's own tokens can never match what it is about to copy. Capped: the last few messages, 8k tokens
/// each and 16k in total, because this is drafted from, not attended to.
fn draft_sources(st: &AppState, messages: &Value) -> Vec<Vec<u32>> {
    const PER: usize = 8192;
    const TOTAL: usize = 16384;
    let mut out: Vec<Vec<u32>> = vec![];
    let mut left = TOTAL;
    let Some(msgs) = messages.as_array() else { return out };
    for m in msgs.iter().rev().take(6) {
        if left == 0 {
            break;
        }
        let text = match &m["content"] {
            Value::String(s) => s.clone(),
            Value::Array(a) => a
                .iter()
                .filter_map(|p| p["text"].as_str())
                .collect::<Vec<_>>()
                .join("\n"),
            _ => continue,
        };
        let norm = original_text(&text);
        if norm.trim().is_empty() {
            continue;
        }
        let Ok(mut ids) = st.chat.encode(&norm) else { continue };
        ids.truncate(left.min(PER));
        if ids.len() > 8 {
            left -= ids.len();
            out.push(ids);
        }
    }
    out
}

/// The text a tool result stands for: JSON escapes resolved, then a leading `NNN→` / `NNN:` / `NNN\t` / `NNN| ` line
/// prefix removed (the forms `cat -n`, `sed =`, GitHub and the usual editors produce). Anything else is left alone,
/// and a wrong reduction only costs drafts, never correctness: every draft is verified against the model's logits.
fn original_text(s: &str) -> String {
    let unescaped = if s.contains("\\n") || s.contains("\\\"") || s.contains("\\u") {
        unescape(s)
    } else {
        s.to_string()
    };
    let mut out = String::with_capacity(unescaped.len());
    for line in unescaped.split_inclusive('\n') {
        let (body, nl) = match line.strip_suffix('\n') {
            Some(b) => (b, "\n"),
            None => (line, ""),
        };
        let digits = body.trim_start();
        let n = digits.len() - digits.trim_start_matches(|c: char| c.is_ascii_digit()).len();
        let rest = &digits[n..];
        let sep = rest.starts_with('→')
            || rest.starts_with('|')
            || rest.starts_with(':')
            || rest.starts_with('\t')
            || (rest.starts_with(' ') && n > 0);
        if n > 0 && n <= 6 && sep {
            let rest = rest.trim_start_matches(['→', '|', ':', '\t', ' ']);
            out.push_str(rest);
        } else {
            out.push_str(body);
        }
        out.push_str(nl);
    }
    out
}

/// JSON string escapes resolved (`\n`, `\t`, `\r`, `\"`, `\\` and `\uXXXX`); an unknown escape is kept verbatim.
fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('"') => out.push('"'),
            Some('\\') => out.push('\\'),
            Some('/') => out.push('/'),
            Some('u') => {
                let hex: String = it.by_ref().take(4).collect();
                match u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
                    Some(c) => out.push(c),
                    None => {
                        out.push_str("\\u");
                        out.push_str(&hex);
                    }
                }
            }
            Some(o) => {
                out.push('\\');
                out.push(o);
            }
            None => out.push('\\'),
        }
    }
    out
}

fn prepare(
    st: &AppState,
    messages: &Value,
    images: Vec<Vec<u8>>,
    tools: Option<&Value>,
    thinking: bool,
    effort: Option<&str>,
    body: &Value,
    max_tokens: Option<u64>,
) -> Result<Prepared, ApiError> {
    let tools = tools.filter(|t| t.as_array().is_some_and(|a| !a.is_empty()));
    let tools_conv = tools.map(|t| {
        Value::Array(
            t.as_array()
                .unwrap()
                .iter()
                .map(|x| {
                    if x.get("function").is_some() {
                        x.clone()
                    } else {
                        json!({ "type": "function", "function": { "name": x["name"], "description": x["description"], "parameters": x["input_schema"] } })
                    }
                })
                .collect(),
        )
    });
    let opts = RenderOptions {
        add_generation_prompt: true,
        enable_thinking: Some(thinking),
        reasoning_effort: effort,
        tools: tools_conv.as_ref(),
    };
    let prompt = st
        .chat
        .render(messages, &opts)
        .map_err(|e| bad(format!("chat template: {e:#}")))?;
    let ids = st.chat.encode(&prompt).map_err(|e| bad(format!("{e:#}")))?;
    // Opt-in: the normalization is correct and wired (a 5-token lookup chain comes out of it, see #25) but at the
    // round depth this engine verifies today it buys nothing, and tokenizing the recent messages costs a few ms on
    // every request. MALLEUS_LOOKUP_NORM=1 turns it on for a measurement.
    let drafts = if std::env::var("MALLEUS_LOOKUP_NORM").is_ok_and(|v| v == "1") {
        draft_sources(st, messages)
    } else {
        vec![]
    };
    if ids.len() >= st.engine.ctx {
        return Err(bad(format!(
            "prompt has {} tokens, context is {}",
            ids.len(),
            st.engine.ctx
        )));
    }
    let image_token = if images.is_empty() {
        None
    } else {
        Some(
            st.chat
                .token_id("<|image_pad|>")
                .ok_or_else(|| bad("tokenizer has no <|image_pad|> token"))?,
        )
    };
    let left = st.engine.ctx - ids.len();
    let max_tokens = (max_tokens.unwrap_or(st.cfg.max_tokens_default as u64) as usize).clamp(1, left);
    let sampling = sampling_from(body, thinking);
    let stops = stops_from(body);
    Ok(Prepared {
        ids,
        drafts,
        images,
        image_token,
        thinking,
        tools: tools.is_some(),
        max_tokens,
        sampling,
        stops,
        logprobs: logprobs(body),
        top_logprobs: top_logprobs(body),
    })
}

/// Sampling parameters the way an OpenAI-style body asks for them; `thinking` picks the temperature/top_p defaults.
fn sampling_from(body: &Value, thinking: bool) -> Sampling {
    let f = |k: &str, d: f64| body[k].as_f64().unwrap_or(d) as f32;
    Sampling {
        temperature: f("temperature", if thinking { 0.6 } else { 0.7 }),
        top_p: f("top_p", if thinking { 0.95 } else { 0.8 }),
        top_k: body["top_k"].as_u64().unwrap_or(20) as usize,
        min_p: f("min_p", 0.0),
        seed: body["seed"].as_u64().unwrap_or_else(now),
    }
}

fn stops_from(body: &Value) -> Vec<String> {
    match body.get("stop").or(body.get("stop_sequences")) {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(String::from)).collect(),
        _ => vec![],
    }
}

/// OpenAI `logprobs` (chat: a boolean; the raw /v1/completions path is parsed separately).
fn logprobs(body: &Value) -> bool {
    let top = top_logprobs(body);
    body["logprobs"].as_bool().unwrap_or(false) || top > 0
}

/// Number of top alternatives requested, capped at 20 like OpenAI.
fn top_logprobs(body: &Value) -> usize {
    body["top_logprobs"].as_u64().unwrap_or(0).min(20) as usize
}

struct CancelGuard(Arc<AtomicBool>);

impl Drop for CancelGuard {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// Decoded generation stream: yields text pieces; stops at stop strings.
struct Gen {
    rx: mpsc::UnboundedReceiver<Event>,
    guard: CancelGuard,
    dec: StreamDecoder,
    text: String,
    stops: Vec<String>,
    stopped_by_string: bool,
    prompt_tokens: usize,
    cached_tokens: usize,
    /// Whether the request asked for per-token log-probabilities.
    logprobs_on: bool,
    /// The log-probability entries of the tokens consumed since the last piece was delivered, in order.
    logprobs: Vec<Logprob>,
}

/// Log-probability data of one generated token (OpenAI `logprobs.content[i]` minus the text, resolved at serialization).
struct Logprob {
    id: u32,
    logprob: f64,
    /// `(id, logprob)` top alternatives, most likely first.
    top: Vec<(u32, f64)>,
}

enum Piece {
    Text {
        text: String,
        /// The log-probability entries of the tokens whose text this piece carries.
        logprobs: Vec<Logprob>,
    },
    Done {
        finish: Finish,
        /// Any log-probability entries not yet delivered with a text piece (generation ended before their text
        /// formed a complete piece, e.g. on a trailing byte-resolution).
        logprobs: Vec<Logprob>,
    },
}

impl Gen {
    fn start(st: &AppState, p: &Prepared) -> Result<Gen, ApiError> {
        let (tx, rx) = mpsc::unbounded_channel();
        let cancel = Arc::new(AtomicBool::new(false));
        st.engine
            .submit(Request {
                prompt: p.ids.clone(),
                drafts: p.drafts.clone(),
                images: p.images.clone(),
                image_token: p.image_token,
                max_tokens: p.max_tokens,
                sampling: p.sampling.clone(),
                stop_tokens: st.eos.clone(),
                logprobs: p.logprobs,
                top_logprobs: p.top_logprobs,
                snapshot_at: st
                    .chat
                    .token_id("<|im_start|>")
                    .and_then(|b| p.ids.iter().rposition(|&t| t == b))
                    .filter(|&i| i > 0),
                cancel: cancel.clone(),
                events: tx,
            })
            .map_err(|e| ApiError(StatusCode::SERVICE_UNAVAILABLE, e.to_string()))?;
        Ok(Gen {
            rx,
            guard: CancelGuard(cancel),
            dec: StreamDecoder::new(),
            text: String::new(),
            stops: p.stops.clone(),
            stopped_by_string: false,
            prompt_tokens: p.ids.len(),
            cached_tokens: 0,
            logprobs_on: p.logprobs,
            logprobs: Vec::new(),
        })
    }

    async fn next(&mut self, chat: &Chat) -> Result<Piece, ApiError> {
        loop {
            match self.rx.recv().await {
                Some(Event::Started {
                    prompt_tokens,
                    cached_tokens,
                }) => {
                    self.prompt_tokens = prompt_tokens;
                    self.cached_tokens = cached_tokens;
                }
                Some(Event::Token { id: t, top }) => {
                    if self.stopped_by_string {
                        continue;
                    }
                    if let Some(p) = top
                        && self.logprobs_on
                    {
                        self.logprobs.push(Logprob {
                            id: t,
                            logprob: p.logprob as f64,
                            top: p.top.iter().map(|&(i, l)| (i, l as f64)).collect(),
                        });
                    }
                    let piece = self
                        .dec
                        .push(chat, t)
                        .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
                    if piece.is_empty() {
                        continue;
                    }
                    let start = self.text.len();
                    self.text.push_str(&piece);
                    for s in &self.stops {
                        let from = start.saturating_sub(s.len());
                        if let Some(i) = self.text[from..].find(s.as_str()) {
                            self.text.truncate(from + i);
                            self.stopped_by_string = true;
                            self.guard.0.store(true, Ordering::Relaxed);
                        }
                    }
                    if self.text.len() > start {
                        return Ok(Piece::Text {
                            text: self.text[start..].to_string(),
                            logprobs: std::mem::take(&mut self.logprobs),
                        });
                    }
                }
                Some(Event::Done(mut f)) => {
                    if self.stopped_by_string {
                        f.reason = FinishReason::Stop;
                    }
                    return Ok(Piece::Done {
                        finish: f,
                        logprobs: std::mem::take(&mut self.logprobs),
                    });
                }
                Some(Event::Error(e)) => return Err(ApiError(StatusCode::INTERNAL_SERVER_ERROR, e)),
                None => return Err(ApiError(StatusCode::INTERNAL_SERVER_ERROR, "engine stopped".into())),
            }
        }
    }
}

/// Incremental splitter of reasoning / content, holding back partial tags and stopping at tool calls.
struct Splitter {
    in_reasoning: bool,
    tools: bool,
    in_tool: bool,
    content_started: bool,
    buf: String,
}

#[derive(Debug)]
enum Delta {
    Reasoning(String),
    Content(String),
}

impl Splitter {
    fn new(thinking: bool, tools: bool) -> Self {
        Splitter {
            in_reasoning: thinking,
            tools,
            in_tool: false,
            content_started: false,
            buf: String::new(),
        }
    }

    fn push(&mut self, s: &str, last: bool) -> Vec<Delta> {
        self.buf.push_str(s);
        let mut out = Vec::new();
        loop {
            if self.in_tool {
                self.buf.clear();
                return out;
            }
            if self.in_reasoning {
                if let Some(i) = self.buf.find(THINK_END) {
                    let r = self.buf[..i].to_string();
                    if !r.is_empty() {
                        out.push(Delta::Reasoning(r));
                    }
                    self.buf.drain(..i + THINK_END.len());
                    self.in_reasoning = false;
                    continue;
                }
                let keep = if last { 0 } else { holdback(&self.buf, &[THINK_END]) };
                let n = self.buf.len() - keep;
                if n > 0 {
                    out.push(Delta::Reasoning(self.buf[..n].to_string()));
                    self.buf.drain(..n);
                }
                return out;
            }
            if !self.content_started {
                let t = self.buf.trim_start().len();
                self.buf.drain(..self.buf.len() - t);
                if self.buf.is_empty() {
                    return out;
                }
                self.content_started = true;
            }
            if self.tools
                && let Some(i) = self.buf.find(TOOL_START)
            {
                let c = self.buf[..i].trim_end().to_string();
                if !c.is_empty() {
                    out.push(Delta::Content(c));
                }
                self.in_tool = true;
                continue;
            }
            let keep = if last || !self.tools {
                0
            } else {
                holdback(&self.buf, &[TOOL_START])
            };
            let n = self.buf.len() - keep;
            if n > 0 {
                out.push(Delta::Content(self.buf[..n].to_string()));
                self.buf.drain(..n);
            }
            return out;
        }
    }
}

fn thinking_flag(st: &AppState, body: &Value) -> bool {
    body["chat_template_kwargs"]["enable_thinking"]
        .as_bool()
        .or(body["enable_thinking"].as_bool())
        .unwrap_or(st.cfg.default_thinking)
}

fn effort(body: &Value) -> Option<&str> {
    body["chat_template_kwargs"]["reasoning_effort"]
        .as_str()
        .or(body["reasoning_effort"].as_str())
        .or(body["reasoning"]["effort"].as_str())
        .map(|e| match e {
            "high" | "xhigh" | "max" => "xhigh",
            "minimal" | "low" => "low",
            _ => "medium",
        })
}

fn finish_openai(f: &Finish, has_tools: bool) -> &'static str {
    match f.reason {
        _ if has_tools => "tool_calls",
        FinishReason::Length => "length",
        _ => "stop",
    }
}

fn usage_openai(f: &Finish, cached: usize) -> Value {
    json!({
        "prompt_tokens": f.prompt_tokens,
        "completion_tokens": f.completion_tokens,
        "total_tokens": f.prompt_tokens + f.completion_tokens,
        "prompt_tokens_details": { "cached_tokens": cached },
    })
}

fn timings(f: &Finish, cached: usize) -> Value {
    let new = f.prompt_tokens.saturating_sub(cached);
    json!({
        "prompt_n": new,
        "prompt_ms": f.prefill_s * 1e3,
        "prompt_per_second": new as f64 / f.prefill_s.max(1e-9),
        "predicted_n": f.completion_tokens,
        "predicted_ms": f.decode_s * 1e3,
        "predicted_per_second": f.completion_tokens as f64 / f.decode_s.max(1e-9),
    })
}

/// One OpenAI logprobs content entry — `{id, token, bytes, logprob, top_logprobs}` — for the token `id` and its
/// top alternatives. The token texts come from a single-token decode; the `bytes` come from the raw BPE piece
/// (byte-level fragments included), the same reconstruction `bench/quality_api.py` builds from the tokenizer, never
/// from re-encoding the decoded text (a split multi-byte fragment would become U+FFFD).
fn logprob_entry(chat: &Chat, id: u32, logprob: f64, top: &[(u32, f64)]) -> Value {
    let token = chat.decode(&[id]).unwrap_or_default();
    let bytes = chat.token_bytes(id).unwrap_or_default();
    let top: Vec<Value> = top
        .iter()
        .map(|&(tid, l)| {
            let t = chat.decode(&[tid]).unwrap_or_default();
            let b = chat.token_bytes(tid).unwrap_or_default();
            json!({ "id": tid, "token": t, "bytes": b, "logprob": l })
        })
        .collect();
    json!({ "id": id, "token": token, "bytes": bytes, "logprob": logprob, "top_logprobs": top })
}

/// The `content` array of OpenAI logprobs for a list of entries.
fn logprobs_value(chat: &Chat, entries: &[Logprob]) -> Value {
    json!({ "content": entries
        .iter()
        .map(|e| logprob_entry(chat, e.id, e.logprob, &e.top))
        .collect::<Vec<_>>() })
}

/// `choices[i].logprobs`: the content array when the request asked for logprobs, null otherwise.
fn response_logprobs(chat: &Chat, on: bool, entries: &[Logprob]) -> Value {
    if !on {
        return Value::Null;
    }
    logprobs_value(chat, entries)
}

async fn chat_completions(
    State(st): State<Shared>,
    h: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, ApiError> {
    check_auth(&st, &h)?;
    let (messages, images) = openai_messages(&body["messages"])?;
    let thinking = thinking_flag(&st, &body);
    let max_tokens = body["max_completion_tokens"].as_u64().or(body["max_tokens"].as_u64());
    let p = prepare(
        &st,
        &messages,
        images,
        body.get("tools"),
        thinking,
        effort(&body),
        &body,
        max_tokens,
    )?;
    let id = format!("chatcmpl-{}-{}", now(), st.counter.fetch_add(1, Ordering::Relaxed));
    let model = st.cfg.model_name.clone();
    let mut g = Gen::start(&st, &p)?;
    if body["stream"].as_bool() != Some(true) {
        let mut logprobs: Vec<Logprob> = Vec::new();
        let f = loop {
            match g.next(&st.chat).await? {
                Piece::Text { logprobs: l, .. } => logprobs.extend(l),
                Piece::Done { finish: f, logprobs: l } => {
                    logprobs.extend(l);
                    break f;
                }
            }
        };
        let parsed = parse_output(&g.text, p.thinking, p.tools);
        let mut msg = json!({ "role": "assistant", "content": parsed.content });
        if !parsed.reasoning.is_empty() {
            msg["reasoning_content"] = json!(parsed.reasoning);
        }
        if !parsed.tool_calls.is_empty() {
            msg["tool_calls"] = openai_tool_calls(&parsed.tool_calls, &id);
        }
        return Ok(Json(json!({
            "id": id, "object": "chat.completion", "created": now(), "model": model,
            "choices": [{ "index": 0, "message": msg,
                "logprobs": response_logprobs(&st.chat, p.logprobs, &logprobs),
                "finish_reason": finish_openai(&f, !parsed.tool_calls.is_empty()) }],
            "usage": usage_openai(&f, g.cached_tokens),
            "timings": timings(&f, g.cached_tokens),
        }))
        .into_response());
    }
    let include_usage = body["stream_options"]["include_usage"].as_bool().unwrap_or(false);
    let stream = openai_stream(st.clone(), g, p, id, model, include_usage);
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response())
}

/// `/v1/completions`: raw text completion, classic OpenAI schema. `logprobs` is an int here — the number of top
/// alternatives per token (0 = none), same as the classic API — and each generated token is reported with the same
/// `{id, token, bytes, logprob, top_logprobs}` shape the chat endpoint uses, so a single tool scores both endpoints.
async fn completions(State(st): State<Shared>, h: HeaderMap, Json(body): Json<Value>) -> Result<Response, ApiError> {
    check_auth(&st, &h)?;
    let prompt = match &body["prompt"] {
        Value::String(s) => s.clone(),
        Value::Array(a) if a.len() == 1 && a[0].is_string() => a[0].as_str().unwrap().to_string(),
        _ => return Err(bad("prompt must be a string (arrays of prompts are not supported)")),
    };
    let ids = st.chat.encode(&prompt).map_err(|e| bad(format!("{e:#}")))?;
    if ids.len() >= st.engine.ctx {
        return Err(bad(format!(
            "prompt has {} tokens, context is {}",
            ids.len(),
            st.engine.ctx
        )));
    }
    let top_logprobs = body["logprobs"].as_u64().unwrap_or(0).min(20) as usize;
    let max_tokens = (body["max_tokens"].as_u64().unwrap_or(st.cfg.max_tokens_default as u64) as usize)
        .clamp(1, st.engine.ctx - ids.len());
    let p = Prepared {
        ids,
        drafts: vec![],
        images: vec![],
        image_token: None,
        thinking: false,
        tools: false,
        max_tokens,
        sampling: sampling_from(&body, false),
        stops: stops_from(&body),
        logprobs: top_logprobs > 0,
        top_logprobs,
    };
    let id = format!("cmpl-{}-{}", now(), st.counter.fetch_add(1, Ordering::Relaxed));
    let model = st.cfg.model_name.clone();
    let mut g = Gen::start(&st, &p)?;
    if body["stream"].as_bool() != Some(true) {
        let mut logprobs: Vec<Logprob> = Vec::new();
        let f = loop {
            match g.next(&st.chat).await? {
                Piece::Text { logprobs: l, .. } => logprobs.extend(l),
                Piece::Done { finish: f, logprobs: l } => {
                    logprobs.extend(l);
                    break f;
                }
            }
        };
        return Ok(Json(json!({
            "id": id, "object": "text_completion", "created": now(), "model": model,
            "choices": [{ "index": 0, "text": g.text,
                "logprobs": response_logprobs(&st.chat, p.logprobs, &logprobs),
                "finish_reason": finish_openai(&f, false) }],
            "usage": usage_openai(&f, g.cached_tokens),
            "timings": timings(&f, g.cached_tokens),
        }))
        .into_response());
    }
    let stream = async_stream(move |tx| async move {
        let send = |v: Value| tx.send(SseEvent::default().data(v.to_string())).is_ok();
        loop {
            let piece = match g.next(&st.chat).await {
                Ok(x) => x,
                Err(e) => {
                    let _ = send(json!({ "error": { "message": e.1 } }));
                    return;
                }
            };
            let (text, lps, done) = match piece {
                Piece::Text { text, logprobs } => (text, logprobs, None),
                Piece::Done { finish, logprobs } => (String::new(), logprobs, Some(finish)),
            };
            let v = json!({ "id": id, "object": "text_completion", "created": now(), "model": model,
                "choices": [{ "index": 0, "text": text, "logprobs": response_logprobs(&st.chat, p.logprobs, &lps),
                    "finish_reason": Value::Null }] });
            if !send(v) {
                return;
            }
            if let Some(f) = done {
                let mut last = json!({ "id": id, "object": "text_completion", "created": now(), "model": model,
                    "choices": [{ "index": 0, "text": "", "logprobs": Value::Null,
                        "finish_reason": finish_openai(&f, false) }] });
                last["timings"] = timings(&f, g.cached_tokens);
                let _ = send(last);
                let _ = send(
                    json!({ "id": id, "object": "text_completion", "created": now(), "model": model,
                    "choices": [], "usage": usage_openai(&f, g.cached_tokens) }),
                );
                let _ = tx.send(SseEvent::default().data("[DONE]"));
                return;
            }
        }
    });
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response())
}

fn openai_stream(
    st: Shared,
    mut g: Gen,
    p: Prepared,
    id: String,
    model: String,
    include_usage: bool,
) -> impl Stream<Item = Result<SseEvent, Infallible>> {
    async_stream(move |tx| async move {
        let chunk = |delta: Value, logprobs: Value, finish: Value| {
            json!({ "id": id, "object": "chat.completion.chunk", "created": now(), "model": model,
                    "choices": [{ "index": 0, "delta": delta, "logprobs": logprobs, "finish_reason": finish }] })
        };
        let send = |v: Value| tx.send(SseEvent::default().data(v.to_string())).is_ok();
        let empty_lp = if p.logprobs {
            json!({ "content": [] })
        } else {
            Value::Null
        };
        if !send(chunk(
            json!({ "role": "assistant", "content": "" }),
            empty_lp.clone(),
            Value::Null,
        )) {
            return;
        }
        let mut sp = Splitter::new(p.thinking, p.tools);
        loop {
            let piece = match g.next(&st.chat).await {
                Ok(x) => x,
                Err(e) => {
                    let _ = send(json!({ "error": { "message": e.1 } }));
                    return;
                }
            };
            let (deltas, done, lps) = match piece {
                Piece::Text { text, logprobs } => (sp.push(&text, false), None, logprobs),
                Piece::Done { finish, logprobs } => (sp.push("", true), Some(finish), logprobs),
            };
            // The entries of the piece go with its first delta chunk; a piece that produced no delta (all text held
            // back) still delivers them, so nothing is dropped.
            let mut attached = false;
            for d in deltas {
                let v = match d {
                    Delta::Reasoning(r) => json!({ "reasoning_content": r }),
                    Delta::Content(c) => json!({ "content": c }),
                };
                let lp = if p.logprobs && !attached {
                    attached = true;
                    logprobs_value(&st.chat, &lps)
                } else {
                    Value::Null
                };
                if !send(chunk(v, lp, Value::Null)) {
                    return;
                }
            }
            if p.logprobs
                && !attached
                && !lps.is_empty()
                && !send(chunk(json!({}), logprobs_value(&st.chat, &lps), Value::Null))
            {
                return;
            }
            if let Some(f) = done {
                let parsed = parse_output(&g.text, p.thinking, p.tools);
                if !parsed.tool_calls.is_empty() {
                    let mut calls = openai_tool_calls(&parsed.tool_calls, &id);
                    for (i, c) in calls.as_array_mut().unwrap().iter_mut().enumerate() {
                        c["index"] = json!(i);
                    }
                    let _ = send(chunk(json!({ "tool_calls": calls }), Value::Null, Value::Null));
                }
                let mut last = chunk(
                    json!({}),
                    Value::Null,
                    json!(finish_openai(&f, !parsed.tool_calls.is_empty())),
                );
                last["timings"] = timings(&f, g.cached_tokens);
                let _ = send(last);
                if include_usage {
                    let _ = send(
                        json!({ "id": id, "object": "chat.completion.chunk", "created": now(), "model": model, "choices": [], "usage": usage_openai(&f, g.cached_tokens) }),
                    );
                }
                let _ = tx.send(SseEvent::default().data("[DONE]"));
                return;
            }
        }
    })
}

/// Minimal channel-backed stream: the producer future runs in a task and ends when the client goes away.
fn async_stream<F, Fut>(f: F) -> impl Stream<Item = Result<SseEvent, Infallible>>
where
    F: FnOnce(mpsc::UnboundedSender<SseEvent>) -> Fut,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    let (tx, rx) = mpsc::unbounded_channel();
    let task = tokio::spawn(f(tx));
    let abort = AbortOnDrop(task);
    futures_util::stream::unfold((rx, abort), |(mut rx, abort)| async move {
        rx.recv().await.map(|e| (Ok(e), (rx, abort)))
    })
}

struct AbortOnDrop(tokio::task::JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

fn anthropic_thinking(st: &AppState, body: &Value) -> bool {
    match body["thinking"]["type"].as_str() {
        Some("enabled") | Some("adaptive") => true,
        Some("disabled") => false,
        _ => st.cfg.default_thinking,
    }
}

fn anthropic_stop(f: &Finish, tools: bool, by_string: bool) -> &'static str {
    match f.reason {
        _ if tools => "tool_use",
        FinishReason::Length => "max_tokens",
        _ if by_string => "stop_sequence",
        _ => "end_turn",
    }
}

async fn anthropic_count_tokens(
    State(st): State<Shared>,
    h: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    check_auth(&st, &h)?;
    let (messages, images) = anthropic_messages_to_template(&body)?;
    let p = prepare(
        &st,
        &messages,
        images,
        body.get("tools"),
        anthropic_thinking(&st, &body),
        None,
        &body,
        Some(1),
    )?;
    Ok(Json(json!({ "input_tokens": p.ids.len() })))
}

async fn anthropic_messages(
    State(st): State<Shared>,
    h: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, ApiError> {
    check_auth(&st, &h)?;
    let (messages, images) = anthropic_messages_to_template(&body)?;
    let thinking = anthropic_thinking(&st, &body);
    let p = prepare(
        &st,
        &messages,
        images,
        body.get("tools"),
        thinking,
        effort(&body),
        &body,
        body["max_tokens"].as_u64(),
    )?;
    let id = format!("msg_{}{}", now(), st.counter.fetch_add(1, Ordering::Relaxed));
    let model = st.cfg.model_name.clone();
    let mut g = Gen::start(&st, &p)?;
    if body["stream"].as_bool() != Some(true) {
        let f = loop {
            match g.next(&st.chat).await? {
                Piece::Text { .. } => {}
                Piece::Done { finish: f, .. } => break f,
            }
        };
        let parsed = parse_output(&g.text, p.thinking, p.tools);
        let mut content = Vec::new();
        if !parsed.reasoning.is_empty() {
            content.push(json!({ "type": "thinking", "thinking": parsed.reasoning, "signature": "" }));
        }
        if !parsed.content.is_empty() {
            content.push(json!({ "type": "text", "text": parsed.content }));
        }
        for (i, c) in parsed.tool_calls.iter().enumerate() {
            content.push(
                json!({ "type": "tool_use", "id": format!("toolu_{id}_{i}"), "name": c.name, "input": c.arguments }),
            );
        }
        return Ok(Json(json!({
            "id": id, "type": "message", "role": "assistant", "model": model, "content": content,
            "stop_reason": anthropic_stop(&f, !parsed.tool_calls.is_empty(), g.stopped_by_string), "stop_sequence": null,
            "usage": { "input_tokens": f.prompt_tokens, "output_tokens": f.completion_tokens, "cache_read_input_tokens": g.cached_tokens },
        }))
        .into_response());
    }
    let stream = async_stream(move |tx| async move {
        let ev = |name: &str, v: Value| tx.send(SseEvent::default().event(name).data(v.to_string())).is_ok();
        let start = json!({ "type": "message_start", "message": { "id": id, "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": null, "stop_sequence": null, "usage": { "input_tokens": p.ids.len(), "output_tokens": 0 } } });
        if !ev("message_start", start) {
            return;
        }
        let mut sp = Splitter::new(p.thinking, p.tools);
        let mut idx: i64 = -1;
        let mut cur: Option<&'static str> = None;
        loop {
            let piece = match g.next(&st.chat).await {
                Ok(x) => x,
                Err(e) => {
                    let _ = ev(
                        "error",
                        json!({ "type": "error", "error": { "type": "api_error", "message": e.1 } }),
                    );
                    return;
                }
            };
            let (deltas, done) = match piece {
                Piece::Text { text: t, .. } => (sp.push(&t, false), None),
                Piece::Done { finish: f, .. } => (sp.push("", true), Some(f)),
            };
            for d in deltas {
                let (kind, delta) = match d {
                    Delta::Reasoning(r) => ("thinking", json!({ "type": "thinking_delta", "thinking": r })),
                    Delta::Content(c) => ("text", json!({ "type": "text_delta", "text": c })),
                };
                if cur != Some(kind) {
                    if cur.is_some() {
                        if kind == "text" {
                            let _ = ev(
                                "content_block_delta",
                                json!({ "type": "content_block_delta", "index": idx, "delta": { "type": "signature_delta", "signature": "" } }),
                            );
                        }
                        let _ = ev(
                            "content_block_stop",
                            json!({ "type": "content_block_stop", "index": idx }),
                        );
                    }
                    idx += 1;
                    let block = if kind == "thinking" {
                        json!({ "type": "thinking", "thinking": "", "signature": "" })
                    } else {
                        json!({ "type": "text", "text": "" })
                    };
                    let _ = ev(
                        "content_block_start",
                        json!({ "type": "content_block_start", "index": idx, "content_block": block }),
                    );
                    cur = Some(kind);
                }
                if !ev(
                    "content_block_delta",
                    json!({ "type": "content_block_delta", "index": idx, "delta": delta }),
                ) {
                    return;
                }
            }
            if let Some(f) = done {
                if cur.is_some() {
                    let _ = ev(
                        "content_block_stop",
                        json!({ "type": "content_block_stop", "index": idx }),
                    );
                }
                let parsed = parse_output(&g.text, p.thinking, p.tools);
                for (i, c) in parsed.tool_calls.iter().enumerate() {
                    idx += 1;
                    let _ = ev(
                        "content_block_start",
                        json!({ "type": "content_block_start", "index": idx,
                        "content_block": { "type": "tool_use", "id": format!("toolu_{id}_{i}"), "name": c.name, "input": {} } }),
                    );
                    let _ = ev(
                        "content_block_delta",
                        json!({ "type": "content_block_delta", "index": idx,
                        "delta": { "type": "input_json_delta", "partial_json": c.arguments.to_string() } }),
                    );
                    let _ = ev(
                        "content_block_stop",
                        json!({ "type": "content_block_stop", "index": idx }),
                    );
                }
                let _ = ev(
                    "message_delta",
                    json!({ "type": "message_delta",
                    "delta": { "stop_reason": anthropic_stop(&f, !parsed.tool_calls.is_empty(), g.stopped_by_string), "stop_sequence": null },
                    "usage": { "output_tokens": f.completion_tokens } }),
                );
                let _ = ev("message_stop", json!({ "type": "message_stop" }));
                return;
            }
        }
    });
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitter_streams_reasoning_then_content() {
        let mut s = Splitter::new(true, true);
        let mut all = Vec::new();
        for piece in [
            "I think",
            " so</th",
            "ink>\n\nHello",
            " world<tool",
            "_call>\n<function=x>",
        ] {
            all.extend(s.push(piece, false));
        }
        all.extend(s.push("", true));
        let r: String = all
            .iter()
            .filter_map(|d| {
                if let Delta::Reasoning(r) = d {
                    Some(r.as_str())
                } else {
                    None
                }
            })
            .collect();
        let c: String = all
            .iter()
            .filter_map(|d| {
                if let Delta::Content(c) = d {
                    Some(c.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(r, "I think so");
        assert_eq!(c, "Hello world");
    }

    #[test]
    fn image_parts_are_extracted() {
        let user = json!([{ "type": "text", "text": "hi" },
            { "type": "image_url", "image_url": { "url": "data:image/png;base64,aGVsbG8=" } }]);
        let (content, images) = content_parts(&user, "user").unwrap();
        assert!(content.is_array());
        assert_eq!(images, vec![b"hello".to_vec()]);
        assert!(content_parts(&user, "system").is_err());
        assert!(content_parts(&json!({ "type": "image" }), "user").is_err());

        let msgs = json!([{ "role": "user", "content": [
            { "type": "image_url", "image_url": "data:image/png;base64,aGVsbG8=" },
            { "type": "text", "text": "what is this?" }] }]);
        let (converted, images) = openai_messages(&msgs).unwrap();
        assert_eq!(images.len(), 1);
        assert_eq!(converted[0]["content"][0]["type"], "image_url");
    }

    #[test]
    fn anthropic_image_conversion() {
        let body = json!({ "messages": [{ "role": "user", "content": [
            { "type": "image", "source": { "type": "base64", "media_type": "image/png", "data": "aGVsbG8=" } },
            { "type": "text", "text": "what is this?" }] }] });
        let (m, images) = anthropic_messages_to_template(&body).unwrap();
        assert_eq!(images, vec![b"hello".to_vec()]);
        assert_eq!(m[0]["content"][0]["type"], "image");
        assert_eq!(m[0]["content"][1]["type"], "text");
    }

    #[test]
    fn anthropic_conversion() {
        let body = json!({ "system": "be brief", "messages": [
            { "role": "user", "content": "hi" },
            { "role": "assistant", "content": [{ "type": "tool_use", "id": "t1", "name": "ls", "input": { "path": "." } }] },
            { "role": "user", "content": [{ "type": "tool_result", "tool_use_id": "t1", "content": "a.txt" }] }
        ]});
        let (m, imgs) = anthropic_messages_to_template(&body).unwrap();
        assert!(imgs.is_empty());
        assert_eq!(m[0]["role"], "system");
        assert_eq!(m[2]["tool_calls"][0]["function"]["name"], "ls");
        assert_eq!(m[3]["role"], "tool");
        assert_eq!(m[3]["content"], "a.txt");
    }
}

#[cfg(test)]
mod norm_tests {
    use super::{original_text, unescape};

    #[test]
    fn line_numbers_are_removed() {
        let s = "   1→#include <hip/hip_runtime.h>\n   2→\n  12: let x = 1;\n    5\tif (a) {}\n 7| tail";
        assert_eq!(
            original_text(s),
            "#include <hip/hip_runtime.h>\n\nlet x = 1;\nif (a) {}\ntail"
        );
    }

    #[test]
    fn escapes_are_resolved_before_the_numbers() {
        let s = "  1\\u2192#include <x>\\n  2\\u2192int y;";
        // `\u2192` is the arrow, so the prefix is recognized after the escapes are resolved.
        assert_eq!(original_text(s), "#include <x>\nint y;");
    }

    #[test]
    fn plain_text_and_odd_text_survive() {
        assert_eq!(original_text("ciao\nmondo"), "ciao\nmondo");
        assert_eq!(
            original_text("1234567890 is too long to be a line number"),
            "1234567890 is too long to be a line number"
        );
        assert_eq!(unescape("a\\n\\\"b\\\"\\\\c\\qd"), "a\n\"b\"\\c\\qd");
    }
}
