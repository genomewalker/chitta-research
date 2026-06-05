use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use cr_schema::{MotifMiner, Proposer, SchemaGate, SwitchKT};
use cr_regime::RegimeDetector;
use cr_types::{FitnessCue, NodeId, NodeKind, SchemaRegistry};
use tokio::sync::Mutex;
use tracing::info;

use crate::{Agent, AgentAction, AgentContext};

/// Svayambhu — the self-arising schema agent.
///
/// Watches for FitnessCue triggers from Udgatr (high novelty, low calibration).
/// On trigger: mines the belief graph's edge-type stream for recurring motifs,
/// evaluates them through the MDL gate, and emits ApplySchemaChange on acceptance.
/// Periodically runs resweep() to resurrect previously-rejected proposals.
pub struct Svayambhu {
    gate: Arc<Mutex<SchemaGate<SwitchKT>>>,
    regime: Arc<Mutex<RegimeDetector>>,
    processed_runs: Arc<Mutex<HashSet<NodeId>>>,
    resweep_every: u64,
    step_count: Arc<Mutex<u64>>,
}

impl Svayambhu {
    pub fn new() -> Self {
        Self {
            gate: Arc::new(Mutex::new(SchemaGate::new(SwitchKT::default()))),
            regime: Arc::new(Mutex::new(RegimeDetector::new())),
            processed_runs: Arc::new(Mutex::new(HashSet::new())),
            resweep_every: 10,
            step_count: Arc::new(Mutex::new(0)),
        }
    }
}

impl Default for Svayambhu {
    fn default() -> Self { Self::new() }
}

#[async_trait]
impl Agent for Svayambhu {
    fn name(&self) -> &str { "svayambhu" }

    async fn step(&self, ctx: &AgentContext) -> Result<AgentAction, anyhow::Error> {
        let mut count = self.step_count.lock().await;
        *count += 1;
        let step = *count;
        drop(count);

        let graph = ctx.graph.read().await;
        let events = graph.all_edge_kinds();
        drop(graph);

        let schema = ctx.schema_registry.read().await.clone();
        let mut gate = self.gate.lock().await;

        // Periodic resurrection pass
        if step % self.resweep_every == 0 && !events.is_empty() {
            let resurrected = gate.resweep(&events, &schema);
            if let Some(v) = resurrected.into_iter().next() {
                info!(name = %v.morphism.rationale, delta_bits = v.delta_bits, "svayambhu: resurrected schema type");
                let mut reg = ctx.schema_registry.write().await;
                for op in &v.morphism.ops {
                    if let cr_schema::SchemaOp::AddEdgeType { name, algebra } = op {
                        reg.register(name.clone(), algebra.clone());
                    }
                }
                return Ok(AgentAction::ApplySchemaChange {
                    morphism_json: serde_json::to_string(&v.morphism)?,
                });
            }
        }

        // Find newly scored runs that trigger schema mining
        let graph = ctx.graph.read().await;
        let mut processed = self.processed_runs.lock().await;
        let trigger_runs: Vec<(NodeId, FitnessCue)> = graph.all_nodes().iter()
            .filter_map(|n| {
                if let NodeKind::Run(_) = &n.kind {
                    if let Some(f) = n.fitness {
                        let cue = FitnessCue {
                            run_id: n.id,
                            novelty: f.novelty,
                            calibration_improvement: f.calibration_improvement,
                        };
                        if cue.is_schema_trigger() && !processed.contains(&n.id) {
                            return Some((n.id, cue));
                        }
                    }
                }
                None
            })
            .collect();
        drop(graph);

        if trigger_runs.is_empty() {
            return Ok(AgentAction::Noop);
        }

        // Mark all trigger runs as seen before mining (idempotent)
        for (id, _) in &trigger_runs {
            processed.insert(*id);
        }
        drop(processed);

        if events.len() < 10 {
            // Not enough edge history for meaningful motif mining
            return Ok(AgentAction::Noop);
        }

        let cue = trigger_runs[0].1.clone();
        let proposals = MotifMiner::default().propose(&events, &schema, &cue);

        if proposals.is_empty() {
            return Ok(AgentAction::Noop);
        }

        // Evaluate proposals through MDL gate, take first accepted
        for proposal in proposals {
            let verdict = gate.evaluate(proposal, &events, &schema);
            if verdict.accepted {
                info!(
                    name = %verdict.morphism.rationale,
                    delta_bits = verdict.delta_bits,
                    "svayambhu: accepted new schema type"
                );
                let mut reg = ctx.schema_registry.write().await;
                for op in &verdict.morphism.ops {
                    if let cr_schema::SchemaOp::AddEdgeType { name, algebra } = op {
                        reg.register(name.clone(), algebra.clone());
                    }
                }
                return Ok(AgentAction::ApplySchemaChange {
                    morphism_json: serde_json::to_string(&verdict.morphism)?,
                });
            }
        }

        Ok(AgentAction::Noop)
    }
}
