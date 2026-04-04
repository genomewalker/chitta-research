pub mod room;
pub use room::{DiscussionRoom, DiscussionRoomBuilder, Participant, mock_room, standard_room, three_way_room};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub model: String,
    pub system: String,
    pub messages: Vec<Message>,
    pub max_tokens: u32,
    pub temperature: f32,
}

#[derive(Debug, Clone)]
pub struct CompletionResponse {
    pub content: String,
    pub usage: TokenUsage,
    /// Full debate thread from DiscussionRoom: [(participant_name, message), ...]
    /// Empty for non-room clients.
    pub debate_thread: Vec<(String, String)>,
}

#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    pub input: u64,
    pub output: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("API error ({status}): {body}")]
    Api { status: u16, body: String },
    #[error("retries exhausted after {attempts} attempts")]
    RetriesExhausted { attempts: u32 },
    #[error("process error: {0}")]
    Process(String),
}

/// LLM client that delegates to `claude -p` — no API key needed,
/// reuses the current Claude Code session's authentication.
pub struct ClaudeCliClient {
    pub model: Option<String>,
}

impl ClaudeCliClient {
    pub fn new() -> Self { Self { model: None } }
    pub fn with_model(model: impl Into<String>) -> Self { Self { model: Some(model.into()) } }
}

#[async_trait]
impl LlmClient for ClaudeCliClient {
    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        // Build the full prompt: system + user messages concatenated
        let mut prompt = String::new();
        if !req.system.is_empty() {
            prompt.push_str(&req.system);
            prompt.push_str("\n\n");
        }
        for m in &req.messages {
            if m.role == "user" {
                prompt.push_str(&m.content);
                prompt.push('\n');
            }
        }

        let mut cmd = tokio::process::Command::new("claude");
        cmd.arg("-p")
           .arg("--output-format").arg("text")
           .stdin(std::process::Stdio::piped())
           .stdout(std::process::Stdio::piped())
           .stderr(std::process::Stdio::piped())
           // Prevent Claude Code nested-invocation detection from blocking subprocess calls.
           .env_remove("CLAUDECODE")
           .env_remove("CLAUDE_CODE_ENTRYPOINT");

        if let Some(model) = &self.model {
            cmd.arg("--model").arg(model);
        }

        let mut child = cmd.spawn()
            .map_err(|e| LlmError::Process(e.to_string()))?;

        if let Some(stdin) = child.stdin.take() {
            use tokio::io::AsyncWriteExt;
            let mut stdin = stdin;
            stdin.write_all(prompt.as_bytes()).await
                .map_err(|e| LlmError::Process(e.to_string()))?;
        }

        let output = child.wait_with_output().await
            .map_err(|e| LlmError::Process(e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            return Err(LlmError::Process(format!("claude -p failed: {stderr}")));
        }

        let content = String::from_utf8_lossy(&output.stdout).trim().to_string();
        // claude -p doesn't report token counts; leave at 0
        Ok(CompletionResponse { content, usage: TokenUsage::default(), debate_thread: vec![] })
    }
}

/// LLM client that uses `codex exec` — non-interactive Codex CLI.
/// Model is set via `--config model=<model>`.
pub struct CodexClient {
    pub model: Option<String>,
}

impl CodexClient {
    pub fn new() -> Self { Self { model: None } }
    pub fn with_model(model: impl Into<String>) -> Self { Self { model: Some(model.into()) } }
}

#[async_trait]
impl LlmClient for CodexClient {
    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let mut prompt = String::new();
        if !req.system.is_empty() {
            prompt.push_str(&req.system);
            prompt.push_str("\n\n");
        }
        for m in &req.messages {
            if m.role == "user" {
                prompt.push_str(&m.content);
                prompt.push('\n');
            }
        }

        let mut cmd = tokio::process::Command::new("codex");
        cmd.arg("exec")
           .arg("-")            // read prompt from stdin
           .stdin(std::process::Stdio::piped())
           .stdout(std::process::Stdio::piped())
           .stderr(std::process::Stdio::piped());

        if let Some(model) = &self.model {
            cmd.arg("--config").arg(format!("model=\"{}\"", model));
        }

        let mut child = cmd.spawn()
            .map_err(|e| LlmError::Process(e.to_string()))?;

        if let Some(stdin) = child.stdin.take() {
            use tokio::io::AsyncWriteExt;
            let mut stdin = stdin;
            stdin.write_all(prompt.as_bytes()).await
                .map_err(|e| LlmError::Process(e.to_string()))?;
        }

        let output = child.wait_with_output().await
            .map_err(|e| LlmError::Process(e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            return Err(LlmError::Process(format!("codex exec failed: {stderr}")));
        }

        let content = String::from_utf8_lossy(&output.stdout).trim().to_string();
        Ok(CompletionResponse { content, usage: TokenUsage::default(), debate_thread: vec![] })
    }
}

// ── OpenCode CLI ───────────────────────────────────────────────────────────

/// LLM client that drives `opencode -p` — non-interactive OpenCode CLI.
/// Supports any model OpenCode knows about via `--model`.
/// Multiple instances with different models can coexist as separate room participants.
pub struct OpenCodeClient {
    pub model: Option<String>,
}

impl OpenCodeClient {
    pub fn new() -> Self { Self { model: None } }
    pub fn with_model(model: impl Into<String>) -> Self { Self { model: Some(model.into()) } }
}

#[async_trait]
impl LlmClient for OpenCodeClient {
    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let mut prompt = String::new();
        if !req.system.is_empty() {
            prompt.push_str(&req.system);
            prompt.push_str("\n\n");
        }
        for m in &req.messages {
            if m.role == "user" {
                prompt.push_str(&m.content);
                prompt.push('\n');
            }
        }

        let mut cmd = tokio::process::Command::new("opencode");
        cmd.arg("-p")
           .stdin(std::process::Stdio::piped())
           .stdout(std::process::Stdio::piped())
           .stderr(std::process::Stdio::piped())
           // Don't inherit Claude Code's session env vars
           .env_remove("CLAUDECODE")
           .env_remove("CLAUDE_CODE_ENTRYPOINT");

        if let Some(model) = &self.model {
            cmd.arg("--model").arg(model);
        }

        let mut child = cmd.spawn()
            .map_err(|e| LlmError::Process(e.to_string()))?;

        if let Some(stdin) = child.stdin.take() {
            use tokio::io::AsyncWriteExt;
            let mut stdin = stdin;
            stdin.write_all(prompt.as_bytes()).await
                .map_err(|e| LlmError::Process(e.to_string()))?;
        }

        let output = child.wait_with_output().await
            .map_err(|e| LlmError::Process(e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            return Err(LlmError::Process(format!("opencode -p failed: {stderr}")));
        }

        let content = String::from_utf8_lossy(&output.stdout).trim().to_string();
        Ok(CompletionResponse { content, usage: TokenUsage::default(), debate_thread: vec![] })
    }
}

#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LlmError>;
}

const MAX_RETRIES: u32 = 3;
const RETRYABLE_STATUS: &[u16] = &[429, 503];

async fn retry_complete<F, Fut>(f: F) -> Result<CompletionResponse, LlmError>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<CompletionResponse, LlmError>>,
{
    let mut last_err = None;
    for attempt in 0..MAX_RETRIES {
        match f().await {
            Ok(resp) => return Ok(resp),
            Err(e @ LlmError::Api { status, .. }) if RETRYABLE_STATUS.contains(&status) => {
                let delay = Duration::from_millis(500 * 2u64.pow(attempt));
                tracing::warn!(attempt, ?delay, status, "retryable LLM error, backing off");
                tokio::time::sleep(delay).await;
                last_err = Some(e);
            }
            Err(e) => return Err(e),
        }
    }
    Err(last_err.unwrap_or(LlmError::RetriesExhausted { attempts: MAX_RETRIES }))
}

// ── Anthropic ──────────────────────────────────────────────────────────────

pub struct AnthropicClient {
    client: reqwest::Client,
    api_key: String,
}

impl AnthropicClient {
    pub fn new(api_key: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
        }
    }

    async fn do_complete(&self, req: &CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let body = serde_json::json!({
            "model": req.model,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "system": req.system,
            "messages": req.messages,
        });

        let resp = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = resp.status().as_u16();
        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmError::Api { status, body });
        }

        let data: serde_json::Value = resp.json().await?;
        let content = data["content"][0]["text"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let usage = TokenUsage {
            input: data["usage"]["input_tokens"].as_u64().unwrap_or(0),
            output: data["usage"]["output_tokens"].as_u64().unwrap_or(0),
        };

        Ok(CompletionResponse { content, usage, debate_thread: vec![] })
    }
}

#[async_trait]
impl LlmClient for AnthropicClient {
    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        retry_complete(|| self.do_complete(&req)).await
    }
}

// ── OpenAI-compatible ──────────────────────────────────────────────────────

pub struct OpenAiClient {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl OpenAiClient {
    pub fn new(api_key: String, base_url: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            base_url,
        }
    }

    pub fn openai(api_key: String) -> Self {
        Self::new(api_key, "https://api.openai.com/v1".to_string())
    }

    async fn do_complete(&self, req: &CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let mut messages = Vec::with_capacity(req.messages.len() + 1);
        if !req.system.is_empty() {
            messages.push(serde_json::json!({"role": "system", "content": req.system}));
        }
        for m in &req.messages {
            messages.push(serde_json::json!({"role": m.role, "content": m.content}));
        }

        let body = serde_json::json!({
            "model": req.model,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "messages": messages,
        });

        let resp = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = resp.status().as_u16();
        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmError::Api { status, body });
        }

        let data: serde_json::Value = resp.json().await?;
        let content = data["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let usage = TokenUsage {
            input: data["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
            output: data["usage"]["completion_tokens"].as_u64().unwrap_or(0),
        };

        Ok(CompletionResponse { content, usage, debate_thread: vec![] })
    }
}

#[async_trait]
impl LlmClient for OpenAiClient {
    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        retry_complete(|| self.do_complete(&req)).await
    }
}

// ── Google Gemini ──────────────────────────────────────────────────────────

pub struct GeminiClient {
    client: reqwest::Client,
    api_key: String,
    model: String,
}

impl GeminiClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self { client: reqwest::Client::new(), api_key, model }
    }

    async fn do_complete(&self, req: &CompletionRequest) -> Result<CompletionResponse, LlmError> {
        // Build contents array; system prompt goes into system_instruction
        let mut contents: Vec<serde_json::Value> = Vec::new();
        for m in &req.messages {
            let role = if m.role == "assistant" { "model" } else { "user" };
            contents.push(serde_json::json!({
                "role": role,
                "parts": [{"text": m.content}]
            }));
        }

        let mut body = serde_json::json!({
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": req.max_tokens,
                "temperature": req.temperature,
            }
        });

        if !req.system.is_empty() {
            body["system_instruction"] = serde_json::json!({
                "parts": [{"text": req.system}]
            });
        }

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            self.model, self.api_key
        );

        let resp = self
            .client
            .post(&url)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = resp.status().as_u16();
        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmError::Api { status, body });
        }

        let data: serde_json::Value = resp.json().await?;
        let content = data["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let usage = TokenUsage {
            input: data["usageMetadata"]["promptTokenCount"].as_u64().unwrap_or(0),
            output: data["usageMetadata"]["candidatesTokenCount"].as_u64().unwrap_or(0),
        };

        Ok(CompletionResponse { content, usage, debate_thread: vec![] })
    }
}

#[async_trait]
impl LlmClient for GeminiClient {
    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        retry_complete(|| self.do_complete(&req)).await
    }
}

// ── chitta-bridge exec ────────────────────────────────────────────────────

/// LLM client that delegates to `chitta-bridge --exec`.
///
/// Accepts any backend supported by chitta-bridge: "opencode", "claude", "codex".
/// Optionally reuses a named session for context persistence across calls.
pub struct ChittaBridgeClient {
    pub backend: String,
    pub model: Option<String>,
    pub session_id: Option<String>,
    pub base_url: Option<String>,
}

impl ChittaBridgeClient {
    pub fn new(backend: impl Into<String>) -> Self {
        Self { backend: backend.into(), model: None, session_id: None, base_url: None }
    }

    pub fn with_model(backend: impl Into<String>, model: impl Into<String>) -> Self {
        Self { backend: backend.into(), model: Some(model.into()), session_id: None, base_url: None }
    }

    pub fn with_local(model: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            backend: "local".into(),
            model: Some(model.into()),
            session_id: None,
            base_url: Some(base_url.into()),
        }
    }

    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }
}

#[async_trait]
impl LlmClient for ChittaBridgeClient {
    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let message = req.messages.iter()
            .filter(|m| m.role == "user")
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");

        let mut payload = serde_json::json!({
            "backend": self.backend,
            "system": req.system,
            "message": message,
        });
        if let Some(m) = &self.model {
            payload["model"] = serde_json::Value::String(m.clone());
        }
        if let Some(sid) = &self.session_id {
            payload["session_id"] = serde_json::Value::String(sid.clone());
        }
        if let Some(url) = &self.base_url {
            payload["base_url"] = serde_json::Value::String(url.clone());
        }

        let mut cmd = tokio::process::Command::new("chitta-bridge");
        cmd.arg("--exec")
           .stdin(std::process::Stdio::piped())
           .stdout(std::process::Stdio::piped())
           .stderr(std::process::Stdio::piped());

        let mut child = cmd.spawn()
            .map_err(|e| LlmError::Process(e.to_string()))?;

        if let Some(stdin) = child.stdin.take() {
            use tokio::io::AsyncWriteExt;
            let mut stdin = stdin;
            stdin.write_all(payload.to_string().as_bytes()).await
                .map_err(|e| LlmError::Process(e.to_string()))?;
        }

        let output = child.wait_with_output().await
            .map_err(|e| LlmError::Process(e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            return Err(LlmError::Process(format!("chitta-bridge --exec failed: {stderr}")));
        }

        let data: serde_json::Value = serde_json::from_slice(&output.stdout)
            .map_err(|e| LlmError::Process(format!("invalid JSON from chitta-bridge: {e}")))?;

        if let Some(err) = data["error"].as_str().filter(|s| !s.is_empty()) {
            return Err(LlmError::Process(err.to_string()));
        }

        let content = data["content"].as_str().unwrap_or("").to_string();
        Ok(CompletionResponse { content, usage: TokenUsage::default(), debate_thread: vec![] })
    }
}

/// Ollama client with automatic endpoint discovery.
/// Discovery chain:
/// 1. `/tmp/ollama-server-*.url` cached files (written by chitta-gpu start)
/// 2. `CHITTA_GPU_URL` environment variable
/// 3. SLURM job URL files in `~/.chitta-gpu/jobs/`
/// 4. `http://localhost:11434` (local Ollama)
/// Returns an error if no endpoint is reachable.
pub struct OllamaClient {
    client: reqwest::Client,
    pub model: String,
}

impl OllamaClient {
    pub fn new(model: impl Into<String>) -> Self {
        Self { client: reqwest::Client::new(), model: model.into() }
    }

    async fn discover_endpoint(&self) -> Option<String> {
        // 1. Cached URL files
        if let Ok(entries) = std::fs::read_dir("/tmp") {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with("ollama-server-") && name.ends_with(".url") {
                    if let Ok(url) = std::fs::read_to_string(entry.path()) {
                        let url = url.trim().to_string();
                        if self.probe(&url).await { return Some(url); }
                    }
                }
            }
        }
        // 2. Environment variable
        if let Ok(url) = std::env::var("CHITTA_GPU_URL") {
            if self.probe(&url).await { return Some(url); }
        }
        // 3. SLURM job URL files
        if let Some(home) = std::env::var("HOME").ok() {
            let jobs_dir = format!("{home}/.chitta-gpu/jobs");
            if let Ok(entries) = std::fs::read_dir(&jobs_dir) {
                for entry in entries.flatten() {
                    if let Ok(url) = std::fs::read_to_string(entry.path()) {
                        let url = url.trim().to_string();
                        if self.probe(&url).await { return Some(url); }
                    }
                }
            }
        }
        // 4. Local Ollama
        let local = "http://localhost:11434".to_string();
        if self.probe(&local).await { return Some(local); }
        None
    }

    async fn probe(&self, base_url: &str) -> bool {
        let url = format!("{base_url}/v1/models");
        self.client.get(&url)
            .timeout(Duration::from_secs(2))
            .send().await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }
}

#[async_trait]
impl LlmClient for OllamaClient {
    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let base_url = self.discover_endpoint().await
            .ok_or_else(|| LlmError::Process("No Ollama endpoint reachable (tried cache, env, SLURM, localhost:11434)".into()))?;

        let mut messages = Vec::with_capacity(req.messages.len() + 1);
        if !req.system.is_empty() {
            messages.push(serde_json::json!({"role": "system", "content": req.system}));
        }
        for m in &req.messages {
            messages.push(serde_json::json!({"role": m.role, "content": m.content}));
        }

        let body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "stream": false,
        });

        let resp = self.client
            .post(format!("{base_url}/v1/chat/completions"))
            .header("content-type", "application/json")
            .json(&body)
            .send().await?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmError::Api { status, body });
        }

        let data: serde_json::Value = resp.json().await?;
        let content = data["choices"][0]["message"]["content"]
            .as_str().unwrap_or("").to_string();
        let usage = TokenUsage {
            input: data["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
            output: data["usage"]["completion_tokens"].as_u64().unwrap_or(0),
        };
        Ok(CompletionResponse { content, usage, debate_thread: vec![] })
    }
}

/// Mock LLM client for testing — returns deterministic structured responses
/// so the full pipeline (graph → agents → artifacts) can be tested without
/// any external calls or subprocess spawning.
pub struct MockLlmClient;

impl MockLlmClient {
    pub fn new() -> Self { Self }
}

#[async_trait]
impl LlmClient for MockLlmClient {
    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        // Detect which agent is calling by inspecting the system prompt
        // and the user message content (used for room synthesizer context).
        let system = req.system.to_lowercase();
        let user_content = req.messages.iter()
            .filter(|m| m.role == "user")
            .map(|m| m.content.to_lowercase())
            .collect::<Vec<_>>()
            .join(" ");
        let content = if system.contains("synthesize") || system.contains("synthesizer") {
            // Room synthesizer: infer context from thread content
            if user_content.contains("fitness") || user_content.contains("empirical_gain") || user_content.contains("analyst") || user_content.contains("observations:") || user_content.contains("artifact commit") {
                // Synthesizing for Udgatr
                r#"{
  "novelty": 0.78,
  "empirical_gain": 0.82,
  "reproducibility": 0.71,
  "cost_efficiency": 0.90,
  "transfer_potential": 0.65,
  "calibration_improvement": 0.55,
  "verdict": "confirmed",
  "confidence_update": 0.81,
  "claims": ["Ancient ARGs show lower GC content (p=0.003)", "75% of ARG clades are sister to soil references"]
}"#.to_string()
            } else if user_content.contains("outcome") || user_content.contains("observations") || user_content.contains("executor") || user_content.contains("experiment plan") || user_content.contains("simulate") {
                // Synthesizing for Adhvaryu
                r#"{
  "outcome": "succeeded",
  "observations": ["Identified 47 ARG sequences with >80% identity to CARD", "GC content: ancient 47.2%, modern 52.8% (p=0.003)"],
  "metrics": {"n_args_found": 47, "gc_content_ancient": 47.2, "gc_content_modern": 52.8, "p_value": 0.003, "soil_cluster_fraction": 0.75},
  "summary": "Strong support for GC-content hypothesis. Critic concern about sample size is valid but does not invalidate the trend."
}"#.to_string()
            } else {
                // Synthesizing for Hotr — return hypothesis array
                r#"[
  {
    "statement": "Ancient permafrost ARGs (>10kya) show lower GC content than modern variants due to cold-adaptation pressure",
    "prior_confidence": 0.65,
    "experiment_plan": {
      "description": "Compare GC content of ARG sequences across permafrost age strata vs modern controls",
      "steps": ["Align reads to CARD with minimap2", "Filter by >80% identity", "Compute GC content per contig", "Mann-Whitney U test across strata"]
    }
  }
]"#.to_string()
            }
        } else if system.contains("analyst") && !system.contains("executor") {
            // Udgatr — must check before executor branch since its prompt contains "experiment"
            r#"{
  "novelty": 0.78,
  "empirical_gain": 0.82,
  "reproducibility": 0.71,
  "cost_efficiency": 0.90,
  "transfer_potential": 0.65,
  "calibration_improvement": 0.55,
  "verdict": "confirmed",
  "confidence_update": 0.81,
  "claims": ["Ancient ARGs show lower GC content (p=0.003)", "75% of ARG clades are sister to soil references"]
}"#.to_string()
        } else if system.contains("hypothesis") || system.contains("hotr") {
            r#"[
  {
    "statement": "Ancient permafrost ARGs (>10kya) show lower GC content than modern variants due to cold-adaptation pressure",
    "prior_confidence": 0.65,
    "experiment_plan": {
      "description": "Extract ARG sequences from CARD-matched aDNA reads and compare GC content across age strata",
      "steps": ["Align reads to CARD with minimap2", "Filter by >80% identity", "Compute GC content per contig", "Mann-Whitney U test across strata"]
    }
  },
  {
    "statement": "Permafrost ARGs cluster phylogenetically with soil ARGs rather than clinical isolates, suggesting pre-antibiotic-era origins",
    "prior_confidence": 0.72,
    "experiment_plan": {
      "description": "Build phylogenetic tree of ARG protein sequences from permafrost and CARD reference panel",
      "steps": ["Extract ARG protein sequences", "Align with MUSCLE", "Build ML tree with IQ-TREE2", "Measure clade distances to clinical vs environmental references"]
    }
  }
]"#.to_string()
        } else if system.contains("executor") || system.contains("adhvaryu") || system.contains("experiment") {
            r#"{
  "outcome": "succeeded",
  "observations": [
    "Identified 47 ARG sequences matching CARD with >80% identity across 3 permafrost strata",
    "GC content: 10kya stratum mean=47.2%, modern control mean=52.8% (p=0.003)",
    "ARGs cluster in 4 distinct clades, 3 of which are sister to soil metagenome references"
  ],
  "metrics": {
    "n_args_found": 47,
    "gc_content_ancient": 47.2,
    "gc_content_modern": 52.8,
    "p_value": 0.003,
    "soil_cluster_fraction": 0.75
  },
  "summary": "Found 47 ARGs with significantly lower GC content in ancient permafrost strata (p=0.003), supporting the cold-adaptation hypothesis. 75% of ARG clades are sister to soil references."
}"#.to_string()
        } else if system.contains("udgatr") || system.contains("fitness") {
            // Fallback for other analyst-like prompts
            r#"{
  "novelty": 0.70,
  "empirical_gain": 0.75,
  "reproducibility": 0.65,
  "cost_efficiency": 0.85,
  "transfer_potential": 0.60,
  "calibration_improvement": 0.50,
  "verdict": "confirmed",
  "confidence_update": 0.75,
  "claims": []
}"#.to_string()
        } else {
            "Mock response: task acknowledged.".to_string()
        };

        Ok(CompletionResponse {
            content,
            usage: TokenUsage { input: 100, output: 50 },
            debate_thread: vec![],
        })
    }
}
