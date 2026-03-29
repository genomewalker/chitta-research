/// Researcher agent — pre-populates chitta with web findings for each open question.
///
/// Runs in parallel with Hotr. Before Hotr generates a hypothesis for a question,
/// Researcher searches arXiv, bioRxiv, Semantic Scholar, and GitHub and stores
/// the findings as `kind: wisdom` memories in chitta-field with tag `web-research`.
///
/// Hotr's recall will pick up these memories when they become available, grounding
/// hypotheses in the current state of the literature rather than LLM training alone.

use async_trait::async_trait;
use cr_types::*;
use std::path::PathBuf;

use crate::{Agent, AgentAction, AgentContext};

pub struct Researcher {
    /// Path to the web_research.py script.
    pub script_path: PathBuf,
}

impl Researcher {
    pub fn new() -> Self {
        // Look for the script relative to common install locations
        let candidates = [
            PathBuf::from("scripts/web_research.py"),
            PathBuf::from("/maps/projects/fernandezguerra/apps/repos/chitta-research/scripts/web_research.py"),
            dirs_home().map(|h| h.join(".local/lib/chitta-research/web_research.py"))
                .unwrap_or_default(),
        ];
        let script_path = candidates.into_iter()
            .find(|p| !p.as_os_str().is_empty() && p.exists())
            .unwrap_or_else(|| PathBuf::from("scripts/web_research.py"));
        Self { script_path }
    }
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var("HOME").ok().map(PathBuf::from)
}

/// Track which questions have already been researched (by text hash) to avoid re-fetching.
/// Uses a simple file marker in the artifact dir.
fn already_researched(question: &str, artifacts_dir: &str) -> bool {
    let hash = djb2(question);
    let marker = std::path::Path::new(artifacts_dir).join(format!(".researched-{hash}"));
    marker.exists()
}

fn mark_researched(question: &str, artifacts_dir: &str) {
    let hash = djb2(question);
    let marker = std::path::Path::new(artifacts_dir).join(format!(".researched-{hash}"));
    let _ = std::fs::write(marker, "");
}

fn djb2(s: &str) -> u32 {
    let mut h: u32 = 5381;
    for b in s.bytes() {
        h = h.wrapping_mul(33).wrapping_add(b as u32);
    }
    h
}

#[async_trait]
impl Agent for Researcher {
    fn name(&self) -> &str { "researcher" }

    async fn step(&self, ctx: &AgentContext) -> Result<AgentAction, anyhow::Error> {
        // Only run if chitta is connected — findings need somewhere to go
        let chitta_connected = ctx.chitta.lock().await.connect().await.is_ok();
        if !chitta_connected {
            return Ok(AgentAction::Noop);
        }

        // Find the next unanswered question that hasn't been researched yet
        let question_text = {
            let graph = ctx.graph.read().await;
            let all = graph.all_nodes();

            // Find questions with no hypotheses yet
            let questions: Vec<_> = all.iter()
                .filter(|n| matches!(n.kind, NodeKind::Question(_)))
                .collect();

            let mut target = None;
            for q in &questions {
                let text = match &q.kind {
                    NodeKind::Question(qq) => qq.text.clone(),
                    _ => continue,
                };
                // Check if already researched
                let artifacts_str = ctx.agenda.title.clone(); // use title as dir hint
                if already_researched(&text, "artifacts") {
                    continue;
                }
                // Check if it has hypotheses already (if so, still research, just lower priority)
                let has_hyps = graph.children(q.id, EdgeKind::DerivedFrom)
                    .iter().any(|n| matches!(n.kind, NodeKind::Hypothesis(_)));
                if target.is_none() || !has_hyps {
                    target = Some(text);
                    if !has_hyps { break; } // prefer questions with no hypotheses yet
                }
            }
            target
        };

        let Some(question) = question_text else {
            return Ok(AgentAction::Noop);
        };

        // Run the web research script
        if !self.script_path.exists() {
            tracing::warn!(
                path = %self.script_path.display(),
                "researcher: web_research.py not found, skipping"
            );
            mark_researched(&question, "artifacts");
            return Ok(AgentAction::Noop);
        }

        tracing::info!(question = %&question[..question.len().min(80)], "researcher: fetching web findings");

        let output = tokio::process::Command::new("python3")
            .arg(&self.script_path)
            .arg("--query")
            .arg(&question)
            .arg("--limit")
            .arg("4")
            .output()
            .await?;

        mark_researched(&question, "artifacts");

        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr);
            tracing::warn!(error = %err, "researcher: web_research.py failed");
            return Ok(AgentAction::Noop);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let data: serde_json::Value = serde_json::from_str(&stdout)
            .unwrap_or(serde_json::json!({"results": []}));

        let results = data["results"].as_array().cloned().unwrap_or_default();
        if results.is_empty() {
            tracing::info!("researcher: no results found");
            return Ok(AgentAction::Noop);
        }

        // Store each finding in chitta
        let mut stored = 0usize;
        let mut chitta = ctx.chitta.lock().await;
        for item in &results {
            let source = item["source"].as_str().unwrap_or("web");
            let title  = item["title"].as_str().unwrap_or("").trim();
            let summary = item["summary"].as_str().unwrap_or("").trim();
            let url    = item["url"].as_str().unwrap_or("").trim();
            let year   = item["published"].as_str().unwrap_or("").trim();

            if title.is_empty() && summary.is_empty() { continue; }

            // Format as SSL-compatible wisdom memory
            let content = format!(
                "[web-research:{source}] {title} ({year})\n{summary}\n→ {url}",
                source = source, title = title, year = year,
                summary = &summary[..summary.len().min(400)], url = url
            );

            let _ = chitta.remember(
                &content,
                "wisdom",
                &["web-research", source, "chitta-research"],
                0.70,
            ).await;
            stored += 1;
        }

        tracing::info!(
            stored,
            total = results.len(),
            question = %&question[..question.len().min(60)],
            "researcher: stored web findings"
        );

        Ok(AgentAction::Noop) // Researcher doesn't modify the belief graph directly
    }
}
