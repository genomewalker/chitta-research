use async_trait::async_trait;
use chrono::Utc;
use cr_llm::{CompletionRequest, Message};
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

        let user_msg = format!(
            "Hypothesis: {}\n\nExperiment Plan:\nSteps:\n{}\n\nSimulate executing this experiment and report results.",
            hypothesis_text,
            plan.steps.iter().enumerate().map(|(i, s)| format!("{}. {}", i + 1, s)).collect::<Vec<_>>().join("\n")
        );

        let resp = ctx.llm.complete(CompletionRequest {
            model: String::new(),
            system: SYSTEM_PROMPT.to_string(),
            messages: vec![Message { role: "user".into(), content: user_msg }],
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

        let exec_result: ExecutionResult = serde_json::from_str(json_str)?;

        let status = if exec_result.outcome == "succeeded" {
            RunStatus::Succeeded
        } else {
            RunStatus::Failed
        };

        let cost_usd = (resp.usage.input + resp.usage.output) as f64 * 0.000003;
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
                    llm_tokens_in: resp.usage.input,
                    llm_tokens_out: resp.usage.output,
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
