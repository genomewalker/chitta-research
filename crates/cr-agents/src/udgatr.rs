use async_trait::async_trait;
use cr_llm::{CompletionRequest, Message};
use cr_types::*;

use crate::{Agent, AgentAction, AgentContext};

pub struct Udgatr;

const SYSTEM_PROMPT: &str = r#"You are a scientific analyst. Given an experiment plan and its results, score the findings on these dimensions (each 0.0 to 1.0):

Respond with ONLY valid JSON:
{
  "novelty": 0.0-1.0,
  "empirical_gain": 0.0-1.0,
  "reproducibility": 0.0-1.0,
  "cost_efficiency": 0.0-1.0,
  "transfer_potential": 0.0-1.0,
  "calibration_improvement": 0.0-1.0,
  "verdict": "confirmed" or "refuted" or "inconclusive",
  "confidence_update": 0.0-1.0,
  "claims": ["claim1", "claim2"]
}"#;

#[derive(serde::Deserialize)]
struct AnalysisResult {
    novelty: f32,
    empirical_gain: f32,
    reproducibility: f32,
    cost_efficiency: f32,
    transfer_potential: f32,
    calibration_improvement: f32,
    verdict: String,
    confidence_update: f32,
    #[serde(default)]
    claims: Vec<String>,
}

#[async_trait]
impl Agent for Udgatr {
    fn name(&self) -> &str {
        "udgatr"
    }

    async fn step(&self, ctx: &AgentContext) -> Result<AgentAction, anyhow::Error> {
        let graph = ctx.graph.read().await;
        let all_nodes = graph.all_nodes();

        // Find completed runs that don't have fitness scores yet
        let runs_without_fitness: Vec<_> = all_nodes
            .iter()
            .filter_map(|n| {
                if n.fitness.is_some() {
                    return None;
                }
                match &n.kind {
                    NodeKind::Run(r) if r.status == RunStatus::Succeeded => Some((n.id, r.clone())),
                    _ => None,
                }
            })
            .collect();

        let Some((run_id, run)) = runs_without_fitness.first() else {
            return Ok(AgentAction::Noop);
        };
        let run_id = *run_id;
        let run = run.clone();

        // Gather context: plan + observations
        let plan_node = graph.get_node(run.plan_id);
        let plan_text = plan_node.map(|n| match &n.kind {
            NodeKind::ExperimentPlan(p) => format!("Steps: {}", p.steps.join("; ")),
            _ => String::new(),
        }).unwrap_or_default();

        let hypothesis_id = plan_node.and_then(|n| match &n.kind {
            NodeKind::ExperimentPlan(p) => Some(p.hypothesis_id),
            _ => None,
        });

        let hypothesis_text = hypothesis_id.and_then(|hid| graph.get_node(hid)).map(|n| match &n.kind {
            NodeKind::Hypothesis(h) => h.statement.clone(),
            _ => String::new(),
        }).unwrap_or_default();

        let observations: Vec<String> = all_nodes
            .iter()
            .filter_map(|n| match &n.kind {
                NodeKind::Observation(o) if o.run_id == run_id => Some(o.summary.clone()),
                _ => None,
            })
            .collect();

        drop(graph);

        let user_msg = format!(
            "Hypothesis: {}\n\nExperiment Plan: {}\n\nObservations:\n{}\n\nArtifact commit: {}\n\n\
             REQUIRED OUTPUT FORMAT — respond with ONLY this JSON object:\n\
             {{\n  \"novelty\": 0.0-1.0,\n  \"empirical_gain\": 0.0-1.0,\n\
             \"reproducibility\": 0.0-1.0,\n  \"cost_efficiency\": 0.0-1.0,\n\
             \"transfer_potential\": 0.0-1.0,\n  \"calibration_improvement\": 0.0-1.0,\n\
             \"verdict\": \"confirmed\" or \"refuted\",\n\
             \"confidence_update\": 0.0-1.0,\n  \"claims\": [\"claim1\"]\n}}",
            hypothesis_text,
            plan_text,
            observations.iter().enumerate().map(|(i, o)| format!("{}. {}", i + 1, o)).collect::<Vec<_>>().join("\n"),
            run.artifact_commit.as_deref().unwrap_or("none"),
        );

        let resp = ctx.llm.complete(CompletionRequest {
            model: String::new(),
            system: SYSTEM_PROMPT.to_string(),
            messages: vec![Message { role: "user".into(), content: user_msg }],
            max_tokens: 1024,
            temperature: 0.2,
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

        let analysis: AnalysisResult = serde_json::from_str(json_str)?;

        let fitness = FitnessVector {
            novelty: analysis.novelty,
            empirical_gain: analysis.empirical_gain,
            reproducibility: analysis.reproducibility,
            cost_efficiency: analysis.cost_efficiency,
            transfer_potential: analysis.transfer_potential,
            calibration_improvement: analysis.calibration_improvement,
        };

        let mut graph = ctx.graph.write().await;

        // Update run's fitness
        if let Some(node) = graph.get_node_mut(run_id) {
            node.fitness = Some(fitness);
        }

        // Update hypothesis posterior
        if let Some(h_id) = hypothesis_id {
            if let Some(h_node) = graph.get_node_mut(h_id) {
                if let NodeKind::Hypothesis(ref mut h) = h_node.kind {
                    h.posterior_confidence = Some(analysis.confidence_update);
                }
            }

            // Add claims for strong findings
            if analysis.empirical_gain > 0.6 {
                for claim_text in &analysis.claims {
                    let claim_id = NodeId::new();
                    let claim_node = TypedNode::new(
                        claim_id,
                        NodeKind::Claim(Claim {
                            statement: claim_text.clone(),
                            confidence: analysis.confidence_update,
                            supporting_observations: vec![run_id],
                        }),
                    );
                    graph.add_node(claim_node)?;

                    let edge_kind = match analysis.verdict.as_str() {
                        "confirmed" => EdgeKind::Supports,
                        "refuted" => EdgeKind::Refutes,
                        _ => EdgeKind::Supports,
                    };
                    graph.add_edge(claim_id, h_id, EpistemicEdge {
                        kind: edge_kind,
                        weight: analysis.confidence_update,
                        evidence_ids: vec![run_id],
                    })?;
                }
            }

            // Add triplet to chitta
            if let Ok(mut chitta) = ctx.chitta.try_lock() {
                let predicate = match analysis.verdict.as_str() {
                    "confirmed" => "confirmed_by",
                    "refuted" => "refuted_by",
                    _ => "tested_by",
                };
                let _ = chitta.add_triplet(
                    &format!("hypothesis:{}", h_id),
                    predicate,
                    &format!("run:{}", run_id),
                    analysis.confidence_update,
                ).await;
            }
        }

        tracing::info!(
            run = %run_id,
            verdict = %analysis.verdict,
            empirical_gain = analysis.empirical_gain,
            "udgatr: scored run"
        );

        Ok(AgentAction::ScoreFitness { node_id: run_id, fitness })
    }
}
