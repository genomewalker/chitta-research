use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub Uuid);

impl NodeId {
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }
}

impl Default for NodeId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeKind {
    ResearchProgram(ResearchProgram),
    Question(Question),
    Hypothesis(Hypothesis),
    ExperimentPlan(ExperimentPlan),
    Run(Run),
    Observation(Observation),
    Claim(Claim),
    Method(Method),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchProgram {
    pub title: String,
    pub domain: String,
    pub priority: f32,
    pub max_budget_usd: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Question {
    pub text: String,
    pub program_id: NodeId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hypothesis {
    pub statement: String,
    pub prior_confidence: f32,
    pub posterior_confidence: Option<f32>,
    pub generating_model: String,
    pub tier: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentPlan {
    pub hypothesis_id: NodeId,
    pub steps: Vec<String>,
    pub estimated_cost_usd: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    pub plan_id: NodeId,
    pub status: RunStatus,
    pub started_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
    pub artifact_commit: Option<String>,
    pub resource_usage: ResourceUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub run_id: NodeId,
    pub summary: String,
    pub data_ref: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claim {
    pub statement: String,
    pub confidence: f32,
    pub supporting_observations: Vec<NodeId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Method {
    pub name: String,
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunStatus {
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub gpu_seconds: f64,
    pub cpu_seconds: f64,
    pub llm_tokens_in: u64,
    pub llm_tokens_out: u64,
    pub cost_usd: f64,
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            gpu_seconds: 0.0,
            cpu_seconds: 0.0,
            llm_tokens_in: 0,
            llm_tokens_out: 0,
            cost_usd: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeKind {
    Supports,
    Refutes,
    DerivedFrom,
    GeneralizesTo,
    BlockedBy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicEdge {
    pub kind: EdgeKind,
    pub weight: f32,
    pub evidence_ids: Vec<NodeId>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FitnessVector {
    pub novelty: f32,
    pub empirical_gain: f32,
    pub reproducibility: f32,
    pub cost_efficiency: f32,
    pub transfer_potential: f32,
    pub calibration_improvement: f32,
}

impl FitnessVector {
    pub fn as_array(&self) -> [f32; 6] {
        [
            self.novelty,
            self.empirical_gain,
            self.reproducibility,
            self.cost_efficiency,
            self.transfer_potential,
            self.calibration_improvement,
        ]
    }

    pub fn dominates(&self, other: &FitnessVector) -> bool {
        let a = self.as_array();
        let b = other.as_array();
        let mut strictly_better = false;
        for (ai, bi) in a.iter().zip(b.iter()) {
            if ai < bi {
                return false;
            }
            if ai > bi {
                strictly_better = true;
            }
        }
        strictly_better
    }

    pub fn weighted_scalar(&self, weights: &[f32; 6]) -> f32 {
        let a = self.as_array();
        a.iter().zip(weights.iter()).map(|(v, w)| v * w).sum()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypedNode {
    pub id: NodeId,
    pub kind: NodeKind,
    pub created_at: DateTime<Utc>,
    pub fitness: Option<FitnessVector>,
    pub chitta_memory_id: Option<u64>,
}

impl TypedNode {
    pub fn new(id: NodeId, kind: NodeKind) -> Self {
        Self {
            id,
            kind,
            created_at: Utc::now(),
            fitness: None,
            chitta_memory_id: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_id_uniqueness() {
        let a = NodeId::new();
        let b = NodeId::new();
        assert_ne!(a, b);
    }

    #[test]
    fn fitness_dominates() {
        let a = FitnessVector {
            novelty: 0.8,
            empirical_gain: 0.7,
            reproducibility: 0.9,
            cost_efficiency: 0.6,
            transfer_potential: 0.5,
            calibration_improvement: 0.4,
        };
        let b = FitnessVector {
            novelty: 0.7,
            empirical_gain: 0.6,
            reproducibility: 0.8,
            cost_efficiency: 0.5,
            transfer_potential: 0.4,
            calibration_improvement: 0.3,
        };
        assert!(a.dominates(&b));
        assert!(!b.dominates(&a));
    }

    #[test]
    fn fitness_no_domination_on_tradeoff() {
        let a = FitnessVector {
            novelty: 0.9,
            empirical_gain: 0.3,
            reproducibility: 0.5,
            cost_efficiency: 0.5,
            transfer_potential: 0.5,
            calibration_improvement: 0.5,
        };
        let b = FitnessVector {
            novelty: 0.3,
            empirical_gain: 0.9,
            reproducibility: 0.5,
            cost_efficiency: 0.5,
            transfer_potential: 0.5,
            calibration_improvement: 0.5,
        };
        assert!(!a.dominates(&b));
        assert!(!b.dominates(&a));
    }

    #[test]
    fn fitness_self_does_not_dominate() {
        let a = FitnessVector {
            novelty: 0.5,
            empirical_gain: 0.5,
            reproducibility: 0.5,
            cost_efficiency: 0.5,
            transfer_potential: 0.5,
            calibration_improvement: 0.5,
        };
        assert!(!a.dominates(&a));
    }

    #[test]
    fn weighted_scalar() {
        let f = FitnessVector {
            novelty: 1.0,
            empirical_gain: 0.0,
            reproducibility: 0.0,
            cost_efficiency: 0.0,
            transfer_potential: 0.0,
            calibration_improvement: 0.0,
        };
        let weights = [2.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        assert!((f.weighted_scalar(&weights) - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn serde_roundtrip_node_kind() {
        let h = NodeKind::Hypothesis(Hypothesis {
            statement: "ARGs in permafrost increase with depth".into(),
            prior_confidence: 0.3,
            posterior_confidence: None,
            generating_model: "claude-sonnet-4-6".into(),
            tier: 2,
        });
        let json = serde_json::to_string(&h).unwrap();
        let back: NodeKind = serde_json::from_str(&json).unwrap();
        match back {
            NodeKind::Hypothesis(hyp) => {
                assert_eq!(hyp.statement, "ARGs in permafrost increase with depth");
                assert_eq!(hyp.tier, 2);
            }
            _ => panic!("wrong variant"),
        }
    }
}
