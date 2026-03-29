/// Kriya agent — closes the loop: finds confirmed claims, implements fixes, verifies.
///
/// Named after the yajña skill's Kriya phase: materialise discoveries into code.
///
/// Promotion controller loop:
/// 1. Find the highest-confidence confirmed Claim not yet applied
/// 2. Ask the LLM to generate a concrete fix (shell command or patch)
/// 3. Apply the fix
/// 4. Run the verifier:
///    - If agenda supplies a `VerifierSpec` → run its `cmd`, parse JSON `VerificationResult`
///    - Otherwise → fall back to built-in LOC/test heuristic
/// 5. On `Pass` → commit. On `Fail` → revert + mark attempted.
///    On `Invalid` (build error) → retry up to `build_retries` before giving up.
///    On `Pending` → persist resume_token and return (human-in-the-loop).

use async_trait::async_trait;
use cr_llm::{CompletionRequest, Message};
use cr_types::*;

use crate::{Agent, AgentAction, AgentContext};

const SYSTEM_PROMPT: &str = r#"You are a software engineer applying a research finding to a codebase.

Given a confirmed research claim, produce a concrete fix.
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
  "explanation": "one sentence"
}"#;

fn applied_marker(claim_id: &NodeId) -> String {
    format!(".applied-{}", claim_id)
}

fn extract_file_hint(claim: &str) -> Option<String> {
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

async fn run_cmd(cmd: &str, timeout_secs: u64) -> (i32, String) {
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
            let code = out.status.code().unwrap_or(-1);
            (code, combined[..combined.len().min(2000)].to_string())
        }
        Ok(Err(e)) => (-1, e.to_string()),
        Err(_) => (-1, format!("timed out after {timeout_secs}s")),
    }
}

/// Run the agenda-specified verifier and parse its JSON output into a `VerificationResult`.
/// Distinguishes build failure (exit code in `build_failure_codes`) from hypothesis falsification.
/// Retries on build failure up to `spec.build_retries` times.
async fn run_verifier(spec: &VerifierSpec, output_path: Option<&str>) -> VerificationResult {
    let cmd = if let Some(out) = output_path {
        spec.cmd.replace("{output}", out)
    } else {
        spec.cmd.clone()
    };

    for attempt in 0..=spec.build_retries {
        let (exit_code, raw) = run_cmd(&format!("run: {cmd}"), spec.timeout_s).await;

        if spec.build_failure_codes.contains(&exit_code) {
            if attempt < spec.build_retries {
                tracing::warn!(attempt, exit_code, "verifier: build failure, retrying");
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                continue;
            }
            return VerificationResult {
                status: VerificationStatus::Invalid,
                metrics: Default::default(),
                baseline_metrics: None,
                supports: vec![],
                refutes: vec![],
                cost: None,
                notes: Some(format!("build failure (exit {exit_code}): {}", &raw[..raw.len().min(200)])),
            };
        }

        // Try to parse JSON output from verifier
        if let Some(json_start) = raw.find('{') {
            if let Some(json_end) = raw.rfind('}') {
                let json_str = &raw[json_start..=json_end];
                if let Ok(result) = serde_json::from_str::<VerificationResult>(json_str) {
                    return result;
                }
            }
        }

        // Verifier didn't emit structured JSON — treat exit 0 as pass, else fail
        let status = if exit_code == 0 {
            VerificationStatus::Pass
        } else {
            VerificationStatus::Fail
        };
        return VerificationResult {
            status,
            metrics: Default::default(),
            baseline_metrics: None,
            supports: vec![],
            refutes: vec![],
            cost: None,
            notes: Some(format!("raw output: {}", &raw[..raw.len().min(300)])),
        };
    }

    VerificationResult {
        status: VerificationStatus::Invalid,
        metrics: Default::default(),
        baseline_metrics: None,
        supports: vec![],
        refutes: vec![],
        cost: None,
        notes: Some("max build retries exceeded".to_string()),
    }
}

/// Fallback verifier: built-in LOC/test/warning heuristic for code-only claims.
async fn run_builtin_verifier(claim_text: &str) -> (String, String, i64) {
    let claim_lower = claim_text.to_lowercase();
    let (baseline_cmd, verifier_desc) = if claim_lower.contains("test") || claim_lower.contains("panic") || claim_lower.contains("error") {
        ("cd /maps/projects/fernandezguerra/apps/repos/chitta-research && CARGO_TARGET_DIR=/tmp/cr-target ./build.sh test 2>&1 | grep -E '^test result' | grep -o '[0-9]* failed' | awk '{print $1}' || echo 0",
         "cargo test failures (lower=better)")
    } else if claim_lower.contains("warning") || claim_lower.contains("lint") || claim_lower.contains("unused") {
        ("cd /maps/projects/fernandezguerra/apps/repos/chitta-research && CARGO_TARGET_DIR=/tmp/cr-target ./build.sh build 2>&1 | grep -c '^warning' || echo 0",
         "compiler warnings (lower=better)")
    } else {
        ("wc -l /maps/projects/fernandezguerra/apps/repos/chitta-research/crates/cr-agents/src/*.rs | grep total | awk '{print $1}'",
         "total LOC in cr-agents (lower or unchanged=ok)")
    };
    let (_, out) = run_cmd(&format!("run: {baseline_cmd}"), 120).await;
    let count: i64 = out.lines().last().and_then(|l| l.trim().parse().ok()).unwrap_or(0);
    (baseline_cmd.to_string(), verifier_desc.to_string(), count)
}

pub struct Kriya;

#[async_trait]
impl Agent for Kriya {
    fn name(&self) -> &str { "kriya" }

    async fn step(&self, ctx: &AgentContext) -> Result<AgentAction, anyhow::Error> {
        let (claim_id, claim_text, claim_conf) = {
            let graph = ctx.graph.read().await;
            let mut best: Option<(NodeId, String, f32)> = None;
            for n in graph.all_nodes() {
                if let NodeKind::Claim(c) = &n.kind {
                    if c.confidence < 0.7 { continue; }
                    if is_applied(&n.id) { continue; }
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

        let file_context = if let Some(file_hint) = extract_file_hint(&claim_text) {
            let path = format!("/maps/projects/fernandezguerra/apps/repos/chitta-research/{file_hint}");
            std::fs::read_to_string(&path)
                .map(|s| format!("\nRelevant file ({file_hint}):\n```rust\n{}\n```", &s[..s.len().min(2000)]))
                .unwrap_or_default()
        } else {
            String::new()
        };

        let user_msg = format!(
            "Codebase: /maps/projects/fernandezguerra/apps/repos/chitta-research\n\
             Confirmed claim (confidence {claim_conf:.2}):\n{claim_text}\
             {file_context}\n\n\
             Generate a concrete automated fix. If no safe automated fix exists, \
             output fix_command as 'run: echo skip'.\n\n\
             REQUIRED OUTPUT FORMAT — ONLY JSON, no other text:\n\
             {{\"fix_command\": \"run: <cmd>\", \"explanation\": \"...\"}}",
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

        let fix_cmd = fix_data["fix_command"].as_str().unwrap_or("").to_string();
        let explanation = fix_data["explanation"].as_str().unwrap_or("").to_string();

        if fix_cmd.is_empty() || fix_cmd.contains("manual fix required") || fix_cmd.contains("echo skip") {
            tracing::info!(claim = %&claim_text[..claim_text.len().min(60)], "kriya: no automated fix, skipping");
            mark_applied(&claim_id);
            return Ok(AgentAction::Noop);
        }

        tracing::info!(cmd = %fix_cmd, "kriya: applying fix");
        let (fix_exit, fix_out) = run_cmd(&fix_cmd, 120).await;
        let fix_ok = fix_exit == 0;

        if !fix_ok {
            tracing::warn!(exit_code = fix_exit, output = %&fix_out[..fix_out.len().min(200)], "kriya: fix command failed, reverting");
            let _ = run_cmd("run: cd /maps/projects/fernandezguerra/apps/repos/chitta-research && git checkout .", 15).await;
            mark_applied(&claim_id);
            return Ok(AgentAction::Noop);
        }

        // ── Verify ──────────────────────────────────────────────────────────
        let promoted = if let Some(spec) = &ctx.agenda.verifier {
            let result = run_verifier(spec, None).await;
            tracing::info!(status = ?result.status, metrics = ?result.metrics, "kriya: verifier result");

            match &result.status {
                VerificationStatus::Pass => true,
                VerificationStatus::Fail => {
                    tracing::info!("kriya: verifier fail — reverting");
                    let _ = run_cmd("run: cd /maps/projects/fernandezguerra/apps/repos/chitta-research && git checkout .", 15).await;
                    mark_applied(&claim_id);
                    return Ok(AgentAction::Noop);
                }
                VerificationStatus::Invalid => {
                    // Build/infra error — does not falsify hypothesis; don't mark as attempted
                    tracing::warn!(notes = ?result.notes, "kriya: invalid (build failure) — skipping without penalising hypothesis");
                    let _ = run_cmd("run: cd /maps/projects/fernandezguerra/apps/repos/chitta-research && git checkout .", 15).await;
                    return Ok(AgentAction::Noop);
                }
                VerificationStatus::Pending { resume_token } => {
                    tracing::info!(token = %resume_token, "kriya: awaiting human input — suspending");
                    // Persist resume_token so the daemon can inject results later
                    let _ = std::fs::write(
                        format!(".pending-{}", claim_id),
                        resume_token,
                    );
                    return Ok(AgentAction::Noop);
                }
            }
        } else {
            // Fallback: built-in heuristic verifier
            let (baseline_cmd, verifier_desc, baseline_count) = run_builtin_verifier(&claim_text).await;
            let (_, after_out) = run_cmd(&format!("run: {baseline_cmd}"), 60).await;
            let after_count: i64 = after_out.lines()
                .last().and_then(|l| l.trim().parse().ok()).unwrap_or(999);

            let is_loc_verifier = verifier_desc.contains("LOC");
            let improved = (is_loc_verifier && after_count <= baseline_count)
                || (!is_loc_verifier && after_count < baseline_count)
                || (baseline_count == 0 && fix_ok);

            if !improved {
                tracing::info!(before = baseline_count, after = after_count, "kriya: no improvement — reverting");
                let _ = run_cmd("run: cd /maps/projects/fernandezguerra/apps/repos/chitta-research && git checkout .", 15).await;
                mark_applied(&claim_id);
                return Ok(AgentAction::Noop);
            }
            tracing::info!(before = baseline_count, after = after_count, verifier = verifier_desc, "kriya: improvement confirmed");
            true
        };

        if promoted {
            tracing::info!(explanation = %explanation, "kriya: promoting — committing");
            let (_, _) = run_cmd(
                &format!("run: cd /maps/projects/fernandezguerra/apps/repos/chitta-research && git add -A && git commit -m 'fix(kriya): {explanation}'"),
                30
            ).await;

            mark_applied(&claim_id);

            if let Ok(mut chitta) = ctx.chitta.try_lock() {
                let _ = chitta.remember(
                    &format!("[kriya:applied] {claim_text}\nFix: {fix_cmd}"),
                    "wisdom",
                    &["kriya", "applied-fix", "chitta-research"],
                    0.90,
                ).await;
            }
        }

        Ok(AgentAction::Noop)
    }
}
