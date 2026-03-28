use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

#[derive(Debug, thiserror::Error)]
pub enum ChittaError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("RPC error ({code}): {message}")]
    Rpc { code: i64, message: String },
    #[error("not connected")]
    NotConnected,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallHit {
    pub memory_id: u64,
    pub content: String,
    pub score: f32,
    pub kind: String,
}

fn djb2_hash(s: &str) -> u32 {
    let mut hash: u32 = 5381;
    for c in s.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(c as u32);
    }
    hash
}

fn get_socket_dir() -> PathBuf {
    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        let dir = PathBuf::from(xdg).join("chitta");
        let _ = std::fs::create_dir_all(&dir);
        return dir;
    }
    if let Ok(home) = std::env::var("HOME") {
        let dir = PathBuf::from(home).join(".cache").join("chitta");
        let _ = std::fs::create_dir_all(&dir);
        return dir;
    }
    PathBuf::from("/tmp")
}

pub fn socket_path_for_mind(mind_path: &str) -> PathBuf {
    get_socket_dir().join(format!("chitta-{}.sock", djb2_hash(mind_path)))
}

pub struct ChittaClient {
    socket_path: PathBuf,
    stream: Option<BufReader<UnixStream>>,
    next_id: u64,
}

impl ChittaClient {
    pub fn for_mind(mind_path: &str) -> Self {
        Self {
            socket_path: socket_path_for_mind(mind_path),
            stream: None,
            next_id: 1,
        }
    }

    pub fn from_path(socket_path: &Path) -> Self {
        Self {
            socket_path: socket_path.to_path_buf(),
            stream: None,
            next_id: 1,
        }
    }

    pub async fn connect(&mut self) -> Result<(), ChittaError> {
        let stream = UnixStream::connect(&self.socket_path).await?;
        self.stream = Some(BufReader::new(stream));
        Ok(())
    }

    async fn ensure_connected(&mut self) -> Result<(), ChittaError> {
        if self.stream.is_none() {
            self.connect().await?;
        }
        Ok(())
    }

    pub async fn call(
        &mut self,
        tool: &str,
        args: serde_json::Value,
    ) -> Result<serde_json::Value, ChittaError> {
        self.ensure_connected().await?;

        let id = self.next_id;
        self.next_id += 1;

        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "tools/call",
            "params": {
                "name": tool,
                "arguments": args,
            }
        });

        let mut line = serde_json::to_string(&request)?;
        line.push('\n');

        let reader = self.stream.as_mut().ok_or(ChittaError::NotConnected)?;
        reader.get_mut().write_all(line.as_bytes()).await?;
        reader.get_mut().flush().await?;

        let mut response_line = String::new();
        let n = reader.read_line(&mut response_line).await?;
        if n == 0 {
            self.stream = None;
            return Err(ChittaError::Io(std::io::Error::new(
                std::io::ErrorKind::ConnectionReset,
                "daemon closed connection",
            )));
        }

        let resp: serde_json::Value = serde_json::from_str(&response_line)?;

        if let Some(err) = resp.get("error") {
            return Err(ChittaError::Rpc {
                code: err["code"].as_i64().unwrap_or(-1),
                message: err["message"].as_str().unwrap_or("unknown").to_string(),
            });
        }

        Ok(resp["result"].clone())
    }

    pub async fn remember(
        &mut self,
        content: &str,
        kind: &str,
        tags: &[&str],
        confidence: f32,
    ) -> Result<u64, ChittaError> {
        let args = serde_json::json!({
            "content": content,
            "type": kind,
            "tags": tags,
            "confidence": confidence,
        });
        let result = self.call("remember", args).await?;
        // The daemon returns result with content array; extract memory ID from text
        let text = result["content"][0]["text"]
            .as_str()
            .unwrap_or("");
        // Parse ID from response text like "Remembered #12345 ..."
        let id = text
            .split('#')
            .nth(1)
            .and_then(|s| s.split_whitespace().next())
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        Ok(id)
    }

    pub async fn recall(
        &mut self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<RecallHit>, ChittaError> {
        let args = serde_json::json!({
            "query": query,
            "limit": limit,
        });
        let result = self.call("recall", args).await?;
        let text = result["content"][0]["text"]
            .as_str()
            .unwrap_or("");

        let mut hits = Vec::new();
        for line in text.lines() {
            // Format: "#ID (score) [kind] content..."
            if let Some(rest) = line.strip_prefix('#') {
                let parts: Vec<&str> = rest.splitn(2, ' ').collect();
                if parts.len() < 2 {
                    continue;
                }
                let memory_id = parts[0].parse::<u64>().unwrap_or(0);
                let remainder = parts[1];

                let score = remainder
                    .find('(')
                    .and_then(|start| {
                        remainder[start + 1..]
                            .find(')')
                            .map(|end| &remainder[start + 1..start + 1 + end])
                    })
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.0);

                let kind = remainder
                    .find('[')
                    .and_then(|start| {
                        remainder[start + 1..]
                            .find(']')
                            .map(|end| remainder[start + 1..start + 1 + end].to_string())
                    })
                    .unwrap_or_default();

                let content = remainder
                    .find(']')
                    .map(|i| remainder[i + 1..].trim().to_string())
                    .unwrap_or_default();

                hits.push(RecallHit {
                    memory_id,
                    content,
                    score,
                    kind,
                });
            }
        }
        Ok(hits)
    }

    pub async fn add_triplet(
        &mut self,
        subject: &str,
        predicate: &str,
        object: &str,
        _weight: f32,
    ) -> Result<(), ChittaError> {
        let args = serde_json::json!({
            "subject": subject,
            "predicate": predicate,
            "object": object,
        });
        self.call("connect", args).await?;
        Ok(())
    }

    pub async fn observe(
        &mut self,
        category: &str,
        title: &str,
        content: &str,
    ) -> Result<(), ChittaError> {
        let args = serde_json::json!({
            "category": category,
            "title": title,
            "content": content,
        });
        self.call("observe", args).await?;
        Ok(())
    }
}
