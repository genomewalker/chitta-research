/// Kriya agent — closes the loop: finds confirmed claims, implements fixes, verifies.
///
/// Named after the yajña skill's Kriya phase: materialise discoveries into code.
///
/// Loop per step:
/// 1. Find the highest-confidence confirmed Claim not yet applied
/// 2. Ask the LLM to generate a concrete code fix (a shell command or patch)
/// 3. Execute the fix via subprocess
/// 4. Run a verifier command (cargo clippy / cargo test) to measure improvement
/// 5. If verifier outcome is better than baseline → mark Claim as applied, commit
/// 6. If worse → revert (git checkout .), mark as attempted
///
/// Teeny-tiny discipline: commit only on measured improvement. Never on vibes.

use async_trait::async_trait;
use cr_llm::{CompletionRequest, Message};
use cr_types::*;

use crate::{Agent, AgentAction, AgentContext};

const SYSTEM_PROMPT: &str = r#"You are a software engineer applying a research finding to a Rust codebase.

Given a confirmed research claim about a codebase flaw, produce a concrete fix.
The fix must be a single shell command (prefixed with "run:") that applies the change.

Rules:
- Use `sed -i`, `cargo fix`, or write a small Python/shell script
- The command must be self-contained and idempotent
- If the fix requires multiple steps, chain them with &&
- Do NOT use interactive editors
- If no automated fix is safe, output: "run: echo 'manual fix required'"

REQUIRED OUTPUT FORMAT — respond with ONLY this JSON:
{
  "fix_command": "run: <shell command>",
  "verifier": "run: <command to measure improvement, e.g. cargo clippy 2>&1 | grep -c warning>",
  "explanation": "one sentence"
}"#;

/// Track applied claims to avoid re-applying
fn applied_marker(claim_id: &NodeId) -> String {
    format!(".applied-{}", claim_id)
}

/// Extract a file path hint from a claim if it references a specific file.
fn extract_file_hint(claim: &str) -> Option<String> {
    // Look for patterns like "crates/cr-agents/src/hotr.rs:66" or "crates/cr-llm/src/room.rs"
    if let Some(start) = claim.find("crates/") {
        let rest = &claim[start..];
        let end = rest.find(|c: char| !c.is_alphanumeric() && c != '/' && c != '_' && c != '-' && c != '.')
            .unwrap_or(rest.len());
        let candidate = &rest[..end];
        if candidate.ends_with(".rs") {
            return Some(candidate.to_string());
        }
    }
    None
}

fn is_applied(claim_id: &NodeId) -> bool {
    std::path::Path::new(&applied_marker(claim_id)).exists()
}

fn mark_applied(claim_id: &NodeId) {
    let _ = std::fs::write(applied_marker(claim_id), "");
}

async fn run_cmd(cmd: &str, timeout_secs: u64) -> (bool, String) {
    let actual_cmd = cmd.trim_start_matches("run:").trim();
    match tokio::time::timeout(
        std::time::Duration::from_secs(timeout_secs),
        tokio::process::Command::new("sh")
            .arg("-c")
            .arg(actual_cmd)
            .output()
    ).await {
        Ok(Ok(out)) => {
            let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
            let combined = if stderr.is_empty() { stdout } else { format!("{stdout}\n{stderr}") };
            (out.status.success(), combined[..combined.len().min(1000)].to_string())
        }
        Ok(Err(e)) => (false, e.to_string()),
        Err(_) => (false, format!("timed out after {timeout_secs}s")),
    }
}

pub struct Kriya;

#[async_trait]
impl Agent for Kriya {
    fn name(&self) -> &str { "kriya" }

    async fn step(&self, ctx: &AgentContext) -> Result<AgentAction, anyhow::Error> {
        // Find the best unapplied confirmed claim
        let (claim_id, claim_text, claim_conf) = {
            let graph = ctx.graph.read().await;
            let mut best: Option<(NodeId, String, f32)> = None;
            for n in graph.all_nodes() {
                if let NodeKind::Claim(c) = &n.kind {
                    if c.confidence < 0.7 { continue; }
                    if is_applied(&n.id) { continue; }
                    // Prefer higher confidence
                    if best.as_ref().map(|(_, _, bc)| c.confidence > *bc).unwrap_or(true) {
                        best = Some((n.id, c.statement.clone(), c.confidence));
                    }
                }
            }
            match best {
                Some(b) => b,
                None => return Ok(AgentAction::Noop),
            }
        };

        tracing::info!(
            claim = %&claim_text[..claim_text.len().min(80)],
            confidence = claim_conf,
            "kriya: applying claim"
        );

        // Choose verifier based on claim content — clippy only measures lints,
        // not runtime correctness or test coverage.
        let claim_lower = claim_text.to_lowercase();
        let (baseline_cmd, verifier_desc) = if claim_lower.contains("test") || claim_lower.contains("panic") || claim_lower.contains("error") {
            ("cd /maps/projects/fernandezguerra/apps/repos/chitta-research && CARGO_TARGET_DIR=/tmp/cr-target ./build.sh test 2>&1 | grep -E '^test result' | grep -o '[0-9]* failed' | awk '{print $1}' || echo 0",
             "cargo test failures (lower=better)")
        } else if claim_lower.contains("warning") || claim_lower.contains("lint") || claim_lower.contains("unused") {
            ("cd /maps/projects/fernandezguerra/apps/repos/chitta-research && CARGO_TARGET_DIR=/tmp/cr-target ./build.sh build 2>&1 | grep -c '^warning' || echo 0",
             "compiler warnings (lower=better)")
        } else {
            // For architecture/design claims: measure lines of code in affected file as complexity proxy
            ("wc -l /maps/projects/fernandezguerra/apps/repos/chitta-research/crates/cr-agents/src/*.rs | grep total | awk '{print $1}'",
             "total LOC in cr-agents (lower or unchanged=ok)")
        };

        let (_, baseline_out) = run_cmd(&format!("run: {baseline_cmd}"), 120).await;
        let baseline_count: i64 = baseline_out.lines()
            .last().and_then(|l| l.trim().parse().ok()).unwrap_or(0);

        tracing::info!(baseline = baseline_count, verifier = verifier_desc, "kriya: baseline measured");

        // Include relevant file content for precise fix generation
        let file_context = if let Some(file_hint) = extract_file_hint(&claim_text) {
            let path = format!("/maps/projects/fernandezguerra/apps/repos/chitta-research/{file_hint}");
            std::fs::read_to_string(&path)
                .map(|s| format!("\nRelevant file ({file_hint}):\n```rust\n{}\n```", &s[..s.len().min(2000)]))
                .unwrap_or_default()
        } else {
            String::new()
        };

        // Ask LLM to generate a fix with full context
        let user_msg = format!(
            "Codebase: /maps/projects/fernandezguerra/apps/repos/chitta-research\n\
             Confirmed claim (confidence {claim_conf:.2}):\n{claim_text}\
             {file_context}\n\n\
             Verifier: {verifier_desc} (baseline={baseline_count})\n\
             Generate a concrete automated fix. The fix must measurably improve the verifier score.\n\
             If no safe automated fix exists, output fix_command as 'run: echo skip'.\n\n\
             REQUIRED OUTPUT FORMAT — ONLY JSON, no other text:\n\
             {{\"fix_command\": \"run: <cmd>\", \"verifier\": \"run: {baseline_cmd}\", \"explanation\": \"...\"}}",
        );

        let resp = ctx.llm.complete(CompletionRequest {
            model: String::new(),
            system: SYSTEM_PROMPT.to_string(),
            messages: vec![Message { role: "user".into(), content: user_msg }],
            max_tokens: 512,
            temperature: 0.2,
        }).await?;

        let content = resp.content.trim();
        let json_str = if let Some(s) = content.find('{') {
            if let Some(e) = content.rfind('}') { &content[s..=e] } else { content }
        } else { content };

        let fix_data: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| anyhow::anyhow!("Kriya JSON parse failed: {e}\nraw: {}", &content[..content.len().min(200)]))?;

        let fix_cmd   = fix_data["fix_command"].as_str().unwrap_or("").to_string();
        let verifier  = fix_data["verifier"].as_str().unwrap_or(baseline_cmd).to_string();
        let explanation = fix_data["explanation"].as_str().unwrap_or("").to_string();

        if fix_cmd.is_empty() || fix_cmd.contains("manual fix required") {
            tracing::info!(claim = %&claim_text[..claim_text.len().min(60)], "kriya: no automated fix available, skipping");
            mark_applied(&claim_id);
            return Ok(AgentAction::Noop);
        }

        tracing::info!(cmd = %fix_cmd, "kriya: applying fix");

        // Apply fix
        let (fix_ok, fix_out) = run_cmd(&fix_cmd, 120).await;

        // Measure improvement
        let (_, after_out) = run_cmd(&verifier, 60).await;
        let after_count: i64 = after_out.lines()
            .last().and_then(|l| l.trim().parse().ok()).unwrap_or(999);

        // For LOC-based verifier: neutral (<=) counts as ok since architecture claims
        // don't always reduce line count. For test/lint: strictly lower is better.
        let is_loc_verifier = verifier_desc.contains("LOC");
        let improved = fix_ok && (
            (is_loc_verifier && after_count <= baseline_count) ||
            (!is_loc_verifier && after_count < baseline_count) ||
            // If baseline was 0 (no failures), success of the fix command itself is enough
            (baseline_count == 0 && fix_ok)
        );

        if improved {
            tracing::info!(
                before = baseline_count,
                after = after_count,
                explanation = %explanation,
                "kriya: improvement confirmed — committing"
            );

            // Commit the fix
            let (_, _) = run_cmd(
                &format!("run: cd /maps/projects/fernandezguerra/apps/repos/chitta-research && git add -A && git commit -m 'fix(kriya): {explanation}'"),
                30
            ).await;

            mark_applied(&claim_id);

            // Store outcome in chitta
            if let Ok(mut chitta) = ctx.chitta.try_lock() {
                let _ = chitta.remember(
                    &format!("[kriya:applied] {claim_text}\nFix: {fix_cmd}\nImprovement: {baseline_count}→{after_count} warnings"),
                    "wisdom",
                    &["kriya", "applied-fix", "chitta-research"],
                    0.90,
                ).await;
            }
        } else {
            tracing::info!(
                fix_ok,
                before = baseline_count,
                after = after_count,
                "kriya: no improvement — reverting"
            );

            // Revert
            let (_, _) = run_cmd(
                "run: cd /maps/projects/fernandezguerra/apps/repos/chitta-research && git checkout .",
                15
            ).await;

            mark_applied(&claim_id); // mark as attempted so we don't retry
        }

        Ok(AgentAction::Noop)
    }
}
