use async_trait::async_trait;
use chrono::Utc;
use cr_llm::CompletionRequest;
use cr_types::*;

use crate::{Agent, AgentAction, AgentContext};

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

        let unexecuted: Vec<_> = plans
            .into_iter()
            .filter(|(id, _)| !runs.contains(id))
            .collect();

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

        // Detect subprocess plans — steps starting with "run:" execute shell commands directly.
        // This enables real data analysis: "run: python analyze.py --input data.csv"
        let is_subprocess = plan.steps.iter().any(|s| {
            let s = s.to_lowercase();
            s.starts_with("run:") || s.starts_with("shell:") || s.starts_with("exec:")
        });

        // Detect code-analysis plans — route through chitta tree-sitter instead of LLM.
        let is_code_analysis = !is_subprocess && plan.steps.iter().any(|s| {
            let s = s.to_lowercase();
            s.contains("read") || s.contains("search") || s.contains("symbol") ||
            s.contains("analyze") || s.contains("source") || s.contains("codebase") ||
            s.contains("function") || s.contains("struct") || s.contains("trait")
        });

        let exec_result: ExecutionResult = if is_subprocess {
            execute_subprocess_steps(&plan.steps).await?
        } else if is_code_analysis {
            let mut chitta = ctx.chitta.lock().await;
            let connected = chitta.connect().await.is_ok();
            if connected {
                let mut observations = Vec::new();
                for step in &plan.steps {
                    let sl = step.to_lowercase();
                    let obs = if sl.contains("search") || sl.contains("symbol") {
                        let query = step.split_once(':').map(|(_, q)| q.trim()).unwrap_or(step);
                        chitta.search_symbols(query, 5).await.unwrap_or_else(|e| e.to_string())
                    } else if sl.contains("read") || sl.contains("function") {
                        let name = step.split_once(':').map(|(_, q)| q.trim()).unwrap_or(step);
                        chitta.read_function(name, None).await.unwrap_or_else(|e| e.to_string())
                    } else if sl.contains("codebase") || sl.contains("index") {
                        let path = step.split_once(':').map(|(_, q)| q.trim())
                            .unwrap_or(".");
                        chitta.learn_codebase(path).await.unwrap_or_else(|e| e.to_string())
                    } else {
                        chitta.code_context(step).await.unwrap_or_else(|e| e.to_string())
                    };
                    if !obs.is_empty() && obs.len() < 2000 {
                        observations.push(obs);
                    }
                }
                ExecutionResult {
                    outcome: "succeeded".into(),
                    observations: if observations.is_empty() {
                        vec!["Code analysis completed via tree-sitter.".into()]
                    } else {
                        observations
                    },
                    metrics: serde_json::json!({}),
                    summary: format!("Code analysis via chitta tree-sitter: {} steps executed.", plan.steps.len()),
                    token_usage: (0, 0),
                }
            } else {
                // chitta not connected — fall through to LLM simulation
                execute_via_llm(&ctx, &hypothesis_text, &plan.steps).await?
            }
        } else {
            execute_via_llm(&ctx, &hypothesis_text, &plan.steps).await?
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

        let commit_sha = ctx.artifacts.lock().await.commit_run_artifacts(
            &run_id_str,
            &[("results.json", result_json.as_bytes())],
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

/// Execute plan steps that start with `run:`, `shell:`, or `exec:` as real subprocesses.
/// Captures stdout as observations. Any non-zero exit code marks the run as failed.
/// Steps without a prefix are skipped (treated as documentation).
async fn execute_subprocess_steps(steps: &[String]) -> Result<ExecutionResult, anyhow::Error> {
    let mut observations = Vec::new();
    let mut failed = false;

    for step in steps {
        let cmd = if let Some(c) = step.strip_prefix("run:").or_else(|| step.strip_prefix("shell:")).or_else(|| step.strip_prefix("exec:")) {
            c.trim()
        } else {
            continue; // documentation step, skip
        };

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
                    // Cap observation length — long outputs are truncated
                    let obs = if stdout.len() > 2000 {
                        format!("{}\n[... truncated {} bytes]", &stdout[..2000], stdout.len() - 2000)
                    } else {
                        stdout
                    };
                    observations.push(format!("$ {}\n{}", cmd, obs));
                }

                if !out.status.success() {
                    failed = true;
                    if !stderr.is_empty() {
                        observations.push(format!("stderr: {}", &stderr[..stderr.len().min(500)]));
                    }
                    observations.push(format!("exit code: {}", out.status.code().unwrap_or(-1)));
                }
            }
            Err(e) => {
                failed = true;
                observations.push(format!("failed to spawn '{}': {}", cmd, e));
            }
        }
    }

    if observations.is_empty() {
        observations.push("No subprocess steps produced output.".into());
    }

    Ok(ExecutionResult {
        outcome: if failed { "failed".into() } else { "succeeded".into() },
        observations,
        metrics: serde_json::json!({}),
        summary: format!("{} subprocess steps executed.", steps.iter().filter(|s| {
            let s = s.to_lowercase();
            s.starts_with("run:") || s.starts_with("shell:") || s.starts_with("exec:")
        }).count()),
        token_usage: (0, 0),
    })
}

async fn execute_via_llm(
    ctx: &crate::AgentContext,
    hypothesis_text: &str,
    steps: &[String],
) -> Result<ExecutionResult, anyhow::Error> {
    use cr_llm::CompletionRequest;
    let user_msg = format!(
        "Hypothesis: {}\n\nExperiment Plan:\nSteps:\n{}\n\nSimulate executing this experiment and report results.",
        hypothesis_text,
        steps.iter().enumerate().map(|(i, s)| format!("{}. {}", i + 1, s)).collect::<Vec<_>>().join("\n")
    );

    let resp = ctx.llm.complete(CompletionRequest {
        model: String::new(),
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

    let mut result: ExecutionResult = serde_json::from_str(json_str)?;
    result.token_usage = (resp.usage.input, resp.usage.output);
    Ok(result)
}
