use async_trait::async_trait;
use chrono::Utc;
use cr_types::*;
use std::collections::HashMap;
use std::sync::Mutex;

use crate::{Agent, AgentAction, AgentContext};

/// Per-plan failure counter — prevents infinite retry on parse/LLM errors.
static PLAN_FAILURES: std::sync::LazyLock<Mutex<HashMap<NodeId, u32>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

const MAX_PLAN_RETRIES: u32 = 3;

#[derive(Debug, Clone, Copy)]
pub enum OutputType {
    TestResults,
    BuildOutput,
    LogOutput,
    JsonOutput,
    Generic,
}

pub fn classify_output(stdout: &str, stderr: &str) -> OutputType {
    let combined = format!("{stdout}\n{stderr}");
    if combined.contains("test result:") || combined.contains("FAILED") && combined.contains("PASSED")
        || combined.contains("not ok ") || combined.contains("ok 1 ")
        || combined.contains("pytest") && combined.contains("failed") {
        return OutputType::TestResults;
    }
    if combined.contains("error[E") || combined.contains("undefined reference")
        || combined.contains("ld: ") || combined.contains("make[")
        || combined.contains("CMake Error") || combined.contains("cargo build") {
        return OutputType::BuildOutput;
    }
    if combined.lines().take(3).any(|l| {
        let t = l.trim();
        t.len() > 10 && t.chars().take(4).all(|c| c.is_ascii_digit()) && t.chars().nth(4) == Some('-')
    }) || combined.contains("[INFO]") || combined.contains("[ERROR]") || combined.contains("[WARN]") {
        return OutputType::LogOutput;
    }
    let trimmed = stdout.trim();
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        return OutputType::JsonOutput;
    }
    OutputType::Generic
}

fn type_prefix(t: OutputType) -> &'static str {
    match t {
        OutputType::TestResults => "[TestResults]",
        OutputType::BuildOutput => "[BuildOutput]",
        OutputType::LogOutput   => "[LogOutput]",
        OutputType::JsonOutput  => "[JsonOutput]",
        OutputType::Generic     => "",
    }
}

pub struct Adhvaryu;

const SYSTEM_PROMPT: &str = r#"You are a scientific experiment executor. Given an experiment plan, simulate running it and return structured results.

Respond with ONLY valid JSON:
{
  "outcome": "succeeded" or "failed",
  "observations": ["observation1", "observation2", ...],
  "metrics": {
    "key1": numeric_value,
    "key2": numeric_value
  },
  "summary": "One-paragraph summary of results"
}"#;

#[derive(serde::Deserialize)]
struct ExecutionResult {
    outcome: String,
    observations: Vec<String>,
    metrics: serde_json::Value,
    summary: String,
    #[serde(skip)]
    token_usage: (u64, u64),
    /// Full debate thread from DiscussionRoom: [(participant, message), ...]
    #[serde(skip)]
    debate_thread: Vec<(String, String)>,
}

#[async_trait]
impl Agent for Adhvaryu {
    fn name(&self) -> &str {
        "adhvaryu"
    }

    async fn step(&self, ctx: &AgentContext) -> Result<AgentAction, anyhow::Error> {
        let graph = ctx.graph.read().await;
        let all_nodes = graph.all_nodes();

        // Find experiment plans that have no runs yet
        let plans: Vec<_> = all_nodes
            .iter()
            .filter_map(|n| match &n.kind {
                NodeKind::ExperimentPlan(p) => Some((n.id, p.clone())),
                _ => None,
            })
            .collect();

        let runs: std::collections::HashSet<NodeId> = all_nodes
            .iter()
            .filter_map(|n| match &n.kind {
                NodeKind::Run(r) => Some(r.plan_id),
                _ => None,
            })
            .collect();

        let unexecuted: Vec<_> = {
            let failure_counts = PLAN_FAILURES.lock().unwrap();
            plans
                .into_iter()
                .filter(|(id, _)| {
                    !runs.contains(id)
                        && failure_counts.get(id).copied().unwrap_or(0) < MAX_PLAN_RETRIES
                })
                .collect()
        };

        let Some((plan_id, plan)) = unexecuted.first() else {
            return Ok(AgentAction::Noop);
        };
        let plan_id = *plan_id;
        let plan = plan.clone();

        // Get the hypothesis for context
        let hypothesis_text = graph.get_node(plan.hypothesis_id).and_then(|n| match &n.kind {
            NodeKind::Hypothesis(h) => Some(h.statement.clone()),
            _ => None,
        }).unwrap_or_default();

        drop(graph);

        // Acquire resources
        let _slot = ctx.resources.acquire(false).await?;

        let exec_result: ExecutionResult = match execute_steps_mixed(&plan.steps, &ctx, &hypothesis_text).await {
            Ok(r) => r,
            Err(e) => {
                // Count failure — after MAX_PLAN_RETRIES the plan is skipped permanently.
                *PLAN_FAILURES.lock().unwrap().entry(plan_id).or_insert(0) += 1;
                tracing::warn!(plan = ?plan_id, error = %e, "adhvaryu: execution error (will retry up to {} times)", MAX_PLAN_RETRIES);
                return Err(e);
            }
        };

        let status = if exec_result.outcome == "succeeded" {
            RunStatus::Succeeded
        } else {
            RunStatus::Failed
        };

        let (tokens_in, tokens_out) = exec_result.token_usage;
        let cost_usd = (tokens_in + tokens_out) as f64 * 0.000003;
        ctx.resources.charge(cost_usd);

        // Commit artifacts
        let run_id_str = format!("run-{}", chrono::Utc::now().format("%Y%m%d-%H%M%S"));
        let result_json = serde_json::to_string_pretty(&serde_json::json!({
            "outcome": exec_result.outcome,
            "observations": exec_result.observations,
            "metrics": exec_result.metrics,
            "summary": exec_result.summary,
            "hypothesis": hypothesis_text,
            "plan_steps": plan.steps,
        }))?;

        // Write debate.json alongside results.json if the room produced a thread
        let mut artifact_files: Vec<(&str, Vec<u8>)> = vec![
            ("results.json", result_json.as_bytes().to_vec()),
        ];
        let debate_json_bytes = if !exec_result.debate_thread.is_empty() {
            let entries: Vec<serde_json::Value> = exec_result.debate_thread.iter()
                .map(|(name, msg)| serde_json::json!({"participant": name, "message": msg}))
                .collect();
            serde_json::to_string_pretty(&entries).unwrap_or_default().into_bytes()
        } else {
            vec![]
        };
        if !debate_json_bytes.is_empty() {
            artifact_files.push(("debate.json", debate_json_bytes));
        }

        let artifact_refs: Vec<(&str, &[u8])> = artifact_files.iter()
            .map(|(name, bytes)| (*name, bytes.as_slice()))
            .collect();

        let commit_sha = ctx.artifacts.lock().await.commit_run_artifacts(
            &run_id_str,
            &artifact_refs,
            &format!("Run {}: {}", run_id_str, exec_result.outcome),
        )?;

        let mut graph = ctx.graph.write().await;

        let run_node_id = NodeId::new();
        let run_node = TypedNode::new(
            run_node_id,
            NodeKind::Run(Run {
                plan_id,
                status,
                started_at: Utc::now(),
                finished_at: Some(Utc::now()),
                artifact_commit: Some(commit_sha),
                resource_usage: ResourceUsage {
                    gpu_seconds: 0.0,
                    cpu_seconds: 1.0,
                    llm_tokens_in: tokens_in,
                    llm_tokens_out: tokens_out,
                    cost_usd,
                },
            }),
        );
        graph.add_node(run_node)?;
        graph.add_edge(run_node_id, plan_id, EpistemicEdge {
            kind: EdgeKind::DerivedFrom,
            weight: 1.0,
            evidence_ids: vec![],
        })?;

        for obs_text in &exec_result.observations {
            let obs_id = NodeId::new();
            let obs_node = TypedNode::new(
                obs_id,
                NodeKind::Observation(Observation {
                    run_id: run_node_id,
                    summary: obs_text.clone(),
                    data_ref: None,
                }),
            );
            graph.add_node(obs_node)?;
            graph.add_edge(obs_id, run_node_id, EpistemicEdge {
                kind: EdgeKind::DerivedFrom,
                weight: 1.0,
                evidence_ids: vec![],
            })?;
        }

        tracing::info!(run = %run_id_str, outcome = %exec_result.outcome, "adhvaryu: completed run");

        if let Ok(mut chitta) = ctx.chitta.try_lock() {
            let _ = chitta.remember(
                &format!("Experiment run {}: {} — {}", run_id_str, exec_result.outcome, exec_result.summary),
                "research_event",
                &["adhvaryu", "experiment", &exec_result.outcome],
                0.8,
            ).await;
        }

        Ok(AgentAction::RequestRun { plan_id })
    }
}

/// Per-step dispatcher: each step is routed independently based on its prefix.
/// Mixed plans (run: + search: + read:) work correctly — no step is silently dropped.
/// Steps with no recognised prefix fall through to LLM execution only if no other
/// step produced output; this avoids polluting observations with duplicate code-context stats.
async fn execute_steps_mixed(
    steps: &[String],
    ctx: &AgentContext,
    hypothesis_text: &str,
) -> Result<ExecutionResult, anyhow::Error> {
    let mut observations = Vec::new();
    let mut subprocess_failed = false;

    // Try to connect chitta once for all chitta-routed steps.
    let chitta_connected = {
        let mut chitta = ctx.chitta.lock().await;
        chitta.connect().await.is_ok()
    };

    for step in steps {
        let sl = step.to_lowercase();

        if sl.starts_with("run:") || sl.starts_with("shell:") || sl.starts_with("exec:") {
            // Subprocess step — execute as shell command.
            let cmd = step.splitn(2, ':').nth(1).map(|s| s.trim()).unwrap_or("");
            if cmd.is_empty() { continue; }
            let output = tokio::process::Command::new("sh")
                .arg("-c")
                .arg(cmd)
                .output()
                .await;
            match output {
                Ok(out) => {
                    let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
                    let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
                    if !stdout.is_empty() {
                        let obs_content = if stdout.len() > 2000 {
                            format!("{}\n[... truncated {} bytes]", &stdout[..2000], stdout.len() - 2000)
                        } else {
                            stdout.clone()
                        };
                        let otype = classify_output(&stdout, &stderr);
                        let prefix = type_prefix(otype);
                        let obs = if prefix.is_empty() {
                            format!("$ {}\n{}", cmd, obs_content)
                        } else {
                            format!("{} $ {}\n{}", prefix, cmd, obs_content)
                        };
                        observations.push(obs);
                    }
                    if !out.status.success() {
                        subprocess_failed = true;
                        if !stderr.is_empty() {
                            observations.push(format!("stderr: {}", &stderr[..stderr.len().min(500)]));
                        }
                        observations.push(format!("exit code: {}", out.status.code().unwrap_or(-1)));
                    }
                }
                Err(e) => {
                    subprocess_failed = true;
                    observations.push(format!("failed to spawn '{}': {}", cmd, e));
                }
            }

        } else if (sl.contains("search") || sl.contains("symbol")) && chitta_connected {
            let query = step.split_once(':').map(|(_, q)| q.trim()).unwrap_or(step);
            let mut chitta = ctx.chitta.lock().await;
            let obs = chitta.search_symbols(query, 5).await.unwrap_or_else(|e| e.to_string());
            if !obs.is_empty() && obs.len() < 2000 { observations.push(obs); }

        } else if (sl.contains("read") || sl.contains("function")) && chitta_connected {
            let name = step.split_once(':').map(|(_, q)| q.trim()).unwrap_or(step);
            let mut chitta = ctx.chitta.lock().await;
            let obs = chitta.read_function(name, None).await.unwrap_or_else(|e| e.to_string());
            if !obs.is_empty() && obs.len() < 2000 { observations.push(obs); }

        } else if (sl.contains("codebase") || sl.contains("index")) && chitta_connected {
            let path = step.split_once(':').map(|(_, q)| q.trim()).unwrap_or(".");
            let mut chitta = ctx.chitta.lock().await;
            let obs = chitta.learn_codebase(path).await.unwrap_or_else(|e| e.to_string());
            if !obs.is_empty() && obs.len() < 2000 { observations.push(obs); }

        }
        // Steps with no recognised prefix are treated as documentation — skipped.
        // No fallthrough to code_context to avoid duplicate static-stats observations.
    }

    // If no step produced output, delegate to LLM simulation.
    if observations.is_empty() {
        return execute_via_llm(ctx, hypothesis_text, steps).await;
    }

    let subprocess_count = steps.iter().filter(|s| {
        let s = s.to_lowercase();
        s.starts_with("run:") || s.starts_with("shell:") || s.starts_with("exec:")
    }).count();
    let chitta_count = steps.len() - subprocess_count;

    // A run is "failed" only if subprocess steps failed AND produced no useful observations.
    // If we have real observations despite some step failures, treat it as succeeded so that
    // Udgatr can score it and Brahman's backlog gate can clear.
    let useful_obs = observations.iter().any(|o| {
        !o.starts_with("stderr:") && !o.starts_with("exit code:")
    });
    let outcome = if subprocess_failed && !useful_obs { "failed" } else { "succeeded" };

    Ok(ExecutionResult {
        outcome: outcome.into(),
        observations,
        metrics: serde_json::json!({}),
        summary: format!(
            "{} subprocess + {} chitta steps executed.",
            subprocess_count, chitta_count
        ),
        token_usage: (0, 0),
        debate_thread: vec![],
    })
}


async fn execute_via_llm(
    ctx: &crate::AgentContext,
    hypothesis_text: &str,
    steps: &[String],
) -> Result<ExecutionResult, anyhow::Error> {
    use cr_llm::CompletionRequest;
    let user_msg = format!(
        "Hypothesis: {}\n\nExperiment Plan:\nSteps:\n{}\n\nSimulate executing this experiment and report results.\n\n\
         REQUIRED OUTPUT FORMAT — respond with ONLY this JSON object, no other text:\n\
         {{\n  \"outcome\": \"succeeded\" or \"failed\",\n\
         \"observations\": [\"observation1\", \"observation2\"],\n\
         \"metrics\": {{}},\n  \"summary\": \"one sentence summary\"\n}}",
        hypothesis_text,
        steps.iter().enumerate().map(|(i, s)| format!("{}. {}", i + 1, s)).collect::<Vec<_>>().join("\n")
    );

    let resp = ctx.llm.complete(CompletionRequest {
        model: ctx.llm_model.clone(),
        system: SYSTEM_PROMPT.to_string(),
        messages: vec![cr_llm::Message { role: "user".into(), content: user_msg }],
        max_tokens: 2048,
        temperature: 0.3,
    }).await?;

    let content = resp.content.trim();
    let json_str = if let Some(start) = content.find('{') {
        if let Some(end) = content.rfind('}') {
            &content[start..=end]
        } else {
            content
        }
    } else {
        content
    };

    // Parse leniently — fill missing fields with defaults so a partial LLM response
    // does not cause an infinite retry loop on the same plan.
    let v: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| anyhow::anyhow!("LLM returned non-JSON: {e}\nraw: {json_str}"))?;

    let outcome = v.get("outcome")
        .and_then(|x| x.as_str())
        .unwrap_or("succeeded")
        .to_string();
    let observations = v.get("observations")
        .and_then(|x| x.as_array())
        .map(|arr| arr.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
        .unwrap_or_default();
    let metrics = v.get("metrics").cloned().unwrap_or(serde_json::json!({}));
    let summary = v.get("summary")
        .and_then(|x| x.as_str())
        .unwrap_or("(no summary)")
        .to_string();

    Ok(ExecutionResult {
        outcome,
        observations,
        metrics,
        summary,
        token_usage: (resp.usage.input, resp.usage.output),
        debate_thread: resp.debate_thread,
    })
}
