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
           .stderr(std::process::Stdio::piped());

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
        Ok(CompletionResponse { content, usage: TokenUsage::default() })
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
        Ok(CompletionResponse { content, usage: TokenUsage::default() })
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

        Ok(CompletionResponse { content, usage })
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

        Ok(CompletionResponse { content, usage })
    }
}

#[async_trait]
impl LlmClient for OpenAiClient {
    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        retry_complete(|| self.do_complete(&req)).await
    }
}
