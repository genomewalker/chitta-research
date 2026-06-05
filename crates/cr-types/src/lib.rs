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
    /// Runtime-discovered node type admitted by SchemaGate.
    Custom(CustomNode),
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunStatus {
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
    /// Awaiting human or instrument input before the run can complete.
    AwaitingInput { resume_token: String },
}

/// Spec for an external verifier command, defined per-program in the agenda YAML.
/// The verifier must emit a `VerificationResult` JSON object to stdout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierSpec {
    /// Command to execute. May use `{output}` and `{baseline}` placeholders.
    pub cmd: String,
    /// jq-style path into the JSON output for the primary metric (e.g. ".compression_ratio").
    /// If absent, Udgatr reads `status` directly.
    #[serde(default)]
    pub metric_jsonpath: Option<String>,
    /// Boolean success expression over `metric` and `baseline` (e.g. "metric < baseline * 0.85").
    /// If absent, pass = exit code 0.
    #[serde(default)]
    pub success_expr: Option<String>,
    /// Exit codes that signal build/infra failure (not hypothesis falsification).
    /// Kriya retries on these codes rather than penalising the hypothesis.
    #[serde(default = "default_build_failure_codes")]
    pub build_failure_codes: Vec<i32>,
    /// Max retries on build failure before propagating as `Invalid`.
    #[serde(default = "default_build_retries")]
    pub build_retries: u32,
    /// Verifier command timeout in seconds.
    #[serde(default = "default_verifier_timeout")]
    pub timeout_s: u64,
}

fn default_build_failure_codes() -> Vec<i32> { vec![2] }
fn default_build_retries() -> u32 { 2 }
fn default_verifier_timeout() -> u64 { 300 }

/// Structured output from a verifier command.
/// Verifier scripts must emit exactly one JSON object of this shape to stdout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub status: VerificationStatus,
    /// Named numeric measurements (e.g. `{"compression_ratio": 1.93, "scan_time_ms": 412}`).
    #[serde(default)]
    pub metrics: std::collections::HashMap<String, f64>,
    /// Baseline measurements for comparison.
    #[serde(default)]
    pub baseline_metrics: Option<std::collections::HashMap<String, f64>>,
    /// Claim statements supported by these measurements.
    #[serde(default)]
    pub supports: Vec<String>,
    /// Claim statements refuted by these measurements.
    #[serde(default)]
    pub refutes: Vec<String>,
    /// Estimated cost in USD (optional).
    #[serde(default)]
    pub cost: Option<f64>,
    /// Free-form notes.
    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VerificationStatus {
    Pass,
    Fail,
    /// Awaiting human or instrument input; `resume_token` is used to inject results later.
    Pending { resume_token: String },
    /// Build or infrastructure failure — does not count as hypothesis falsification.
    Invalid,
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

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeKind {
    Supports,
    Refutes,
    DerivedFrom,
    GeneralizesTo,
    BlockedBy,
    /// Runtime-discovered edge type admitted by SchemaGate.
    Custom(String),
}

impl std::fmt::Display for EdgeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EdgeKind::Custom(s) => write!(f, "custom:{s}"),
            other => write!(f, "{other:?}"),
        }
    }
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

// ── Custom node ────────────────────────────────────────────────────────────────

/// A node whose type was discovered at runtime by the schema revision gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomNode {
    pub type_name: String,
    pub content: serde_json::Value,
}

// ── Schema algebra & registry ─────────────────────────────────────────────────

/// Lattice sign for difference-constraint consistency checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeSign {
    /// Monotone-up implication (Supports).
    Up,
    /// Monotone-down implication (Refutes).
    Down,
    /// Equality/propagation (DerivedFrom).
    Eq,
    /// No constraint on the confidence lattice.
    None,
}

/// Structural + lattice metadata for an edge type.
/// Built-in types have default algebras; Custom types register here at accept time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeAlgebra {
    pub acyclic: bool,
    pub transitive: bool,
    pub symmetric: bool,
    pub sign: EdgeSign,
    /// An edge of this type and another of `contradicts` on the same (src, dst)
    /// pair from the same source node constitute a single-source contradiction.
    pub contradicts: Option<String>,
    /// Number of times enforced algebra has been rolled back for this type.
    #[serde(default)]
    pub rollback_count: u32,
}

impl EdgeAlgebra {
    pub fn for_builtin(kind: &EdgeKind) -> Self {
        match kind {
            EdgeKind::Supports     => Self { acyclic: false, transitive: true,  symmetric: false, sign: EdgeSign::Up,   contradicts: Some("Refutes".into()),  rollback_count: 0 },
            EdgeKind::Refutes      => Self { acyclic: false, transitive: false, symmetric: false, sign: EdgeSign::Down, contradicts: Some("Supports".into()), rollback_count: 0 },
            EdgeKind::DerivedFrom  => Self { acyclic: true,  transitive: true,  symmetric: false, sign: EdgeSign::Eq,   contradicts: None,                    rollback_count: 0 },
            EdgeKind::GeneralizesTo=> Self { acyclic: false, transitive: true,  symmetric: false, sign: EdgeSign::None, contradicts: None,                    rollback_count: 0 },
            EdgeKind::BlockedBy    => Self { acyclic: false, transitive: false, symmetric: false, sign: EdgeSign::None, contradicts: None,                    rollback_count: 0 },
            EdgeKind::Custom(_)    => Self { acyclic: false, transitive: false, symmetric: false, sign: EdgeSign::None, contradicts: None,                    rollback_count: 0 },
        }
    }
}

/// Empirical evidence that informed the enforced algebra for a custom edge type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgebraEvidence {
    /// Number of observed instances of this edge type.
    pub support: usize,
    /// Whether a cycle was observed in the corpus.
    pub cycle_observed: bool,
    /// Fraction of (a→b, b→a) pairs relative to total directed pairs.
    pub sym_ratio: f32,
    /// Fraction of (a→b, b→c) ⇒ (a→c) triples that close in the corpus.
    pub trans_closure: f32,
    /// Model confidence that the enforced algebra is correct.
    pub confidence: f64,
}

/// Lifecycle state for a registered edge type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeLifecycle {
    Active,
    Inactive { since_turn: u64 },
    Dead,
}

impl Default for EdgeLifecycle {
    fn default() -> Self { Self::Active }
}

/// Full live certificate for a custom edge type: declared intent vs enforced policy.
/// section_check uses `enforced`; `declared` is the original LLM-proposed algebra.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeTypeSpec {
    pub name: String,
    /// Algebra as declared by the proposing model.
    pub declared: EdgeAlgebra,
    /// Algebra actually enforced by section_check (may be relaxed from declared).
    pub enforced: EdgeAlgebra,
    pub evidence: AlgebraEvidence,
    #[serde(default)]
    pub lifecycle: EdgeLifecycle,
}

/// Which algebra field to mutate in a RelaxAlgebra schema op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlgebraField {
    Acyclic,
    Transitive,
    Symmetric,
}

/// Maps edge type names → EdgeTypeSpec. Built-ins are pre-populated; custom types
/// are inserted when SchemaGate accepts a morphism.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaRegistry {
    algebras: std::collections::HashMap<String, EdgeTypeSpec>,
}

impl SchemaRegistry {
    pub fn new() -> Self {
        let mut algebras = std::collections::HashMap::new();
        for kind in [
            EdgeKind::Supports, EdgeKind::Refutes, EdgeKind::DerivedFrom,
            EdgeKind::GeneralizesTo, EdgeKind::BlockedBy,
        ] {
            let algebra = EdgeAlgebra::for_builtin(&kind);
            let spec = EdgeTypeSpec {
                name: kind.to_string(),
                declared: algebra.clone(),
                enforced: algebra,
                evidence: AlgebraEvidence {
                    support: 0,
                    cycle_observed: false,
                    sym_ratio: 0.0,
                    trans_closure: 0.0,
                    confidence: 1.0,
                },
                lifecycle: EdgeLifecycle::Active,
            };
            algebras.insert(kind.to_string(), spec);
        }
        Self { algebras }
    }

    /// Returns the full spec for a given edge kind.
    pub fn get(&self, kind: &EdgeKind) -> Option<&EdgeTypeSpec> {
        self.algebras.get(&kind.to_string())
    }

    /// Returns only the enforced algebra — use this in section_check, not declared.
    pub fn get_enforced(&self, kind: &EdgeKind) -> Option<&EdgeAlgebra> {
        self.algebras.get(&kind.to_string()).map(|s| &s.enforced)
    }

    /// Register a full EdgeTypeSpec (used when SchemaGate accepts a morphism).
    pub fn register(&mut self, spec: EdgeTypeSpec) {
        self.algebras.insert(spec.name.clone(), spec);
    }

    /// Convenience: register a custom type with declared algebra and permissive
    /// enforced defaults (all structural constraints false). Evidence starts empty.
    pub fn register_algebra(&mut self, name: String, declared: EdgeAlgebra) {
        let enforced = EdgeAlgebra {
            acyclic: false,
            transitive: false,
            symmetric: false,
            sign: declared.sign,
            contradicts: declared.contradicts.clone(),
            rollback_count: 0,
        };
        let spec = EdgeTypeSpec {
            name: name.clone(),
            declared,
            enforced,
            evidence: AlgebraEvidence {
                support: 0,
                cycle_observed: false,
                sym_ratio: 0.0,
                trans_closure: 0.0,
                confidence: 0.0,
            },
            lifecycle: EdgeLifecycle::Active,
        };
        self.algebras.insert(name, spec);
    }

    /// Soft-delete: marks lifecycle as Dead but keeps the spec for provenance.
    pub fn remove(&mut self, name: &str) {
        if let Some(spec) = self.algebras.get_mut(name) {
            spec.lifecycle = EdgeLifecycle::Dead;
        }
    }

    /// Mark as temporarily inactive at `turn` (e.g. not seen in recent corpus window).
    pub fn get_spec_mut(&mut self, name: &str) -> Option<&mut EdgeTypeSpec> {
        self.algebras.get_mut(name)
    }

    pub fn retire(&mut self, name: &str, turn: u64) {
        if let Some(spec) = self.algebras.get_mut(name) {
            spec.lifecycle = EdgeLifecycle::Inactive { since_turn: turn };
        }
    }

    /// Returns specs for Active custom types only (excludes built-ins and Dead/Inactive).
    pub fn custom_edge_specs(&self) -> Vec<&EdgeTypeSpec> {
        self.algebras.values()
            .filter(|s| s.name.starts_with("custom:") && s.lifecycle == EdgeLifecycle::Active)
            .collect()
    }

    /// Legacy helper — names of all registered custom types regardless of lifecycle.
    pub fn known_custom_types(&self) -> Vec<&str> {
        self.algebras.keys()
            .filter(|k| k.starts_with("custom:"))
            .map(|k| k.as_str())
            .collect()
    }
}

impl Default for SchemaRegistry {
    fn default() -> Self { Self::new() }
}

// ── Fitness cue ────────────────────────────────────────────────────────────────

/// Emitted by Udgatr when a scored run shows high novelty but low calibration
/// improvement — the signal to trigger a schema mining pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitnessCue {
    pub run_id: NodeId,
    pub novelty: f32,
    pub calibration_improvement: f32,
}

impl FitnessCue {
    /// Returns true when the fitness pattern suggests a potential new edge/node type.
    pub fn is_schema_trigger(&self) -> bool {
        self.novelty > 0.70 && self.calibration_improvement < 0.30
    }
}

// ── Schema events ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchemaEvent {
    TypeProposed { name: String, delta_bits: f64 },
    TypeAccepted { name: String, delta_bits: f64 },
    TypeRejected { name: String, delta_bits: f64 },
    TypeRetired  { name: String },
    TypesMerged  { kept: String, dropped: String },
    ConsistencyViolation { kind: ConsistencyViolationKind, description: String },
    /// Relax a single structural constraint on the enforced algebra after evidence
    /// shows the declared constraint cannot be maintained without rollback.
    RelaxAlgebra { edge_type: String, field: AlgebraField, new_value: bool },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsistencyViolationKind {
    DerivedFromCycle,
    SingleSourceContradiction,
    StructuralInvariant,
}

// ── Economy of Minds types ─────────────────────────────────────────────────────

pub type LineageId = Uuid;
pub type ActionId = Uuid;

/// The mutable genome of an agent lineage — encodes the miscalibrated thresholds
/// the research findings identified. Mutation acts only on this struct.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Genome {
    pub kriya_confidence_threshold: f64,
    pub udgatr_gain_threshold: f64,
    pub brahman_backlog_limit: usize,
    pub credit_decay_gamma: f64,
    pub hotr_min_events: usize,
}

impl Default for Genome {
    fn default() -> Self {
        Self {
            kriya_confidence_threshold: 0.7,
            udgatr_gain_threshold: 0.35,
            brahman_backlog_limit: 10,
            credit_decay_gamma: 0.7,
            hotr_min_events: 10,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SlotKind { Hypothesis, Experiment, Fix, Synthesis, Schema, Scoring }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LineageStatus { Live, RetiredPendingSettlement }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    pub cash: f64,
    pub reserved: f64,
    pub genome: Genome,
    pub lineage_id: LineageId,
    pub status: LineageStatus,
    pub free_runs_left: u32,
}

impl Account {
    pub fn new(genome: Genome, lineage_id: LineageId, initial_cash: f64) -> Self {
        Self { cash: initial_cash, reserved: 0.0, genome, lineage_id, status: LineageStatus::Live, free_runs_left: 3 }
    }
    pub fn available(&self) -> f64 { (self.cash - self.reserved).max(0.0) }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bid {
    pub lineage_id: LineageId,
    pub slot: SlotKind,
    pub price: f64,
    pub expected_value: f64,
    pub parent_action_ids: Vec<ActionId>,
}

/// Cheap snapshot of system state for pure bid() computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextSnapshot {
    pub untested_backlog: usize,
    pub confirmed_count: usize,
    pub store_density: f64,
    pub budget_remaining_usd: f32,
    pub cycle: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionReceipt {
    pub action_id: ActionId,
    pub lineage_id: LineageId,
    pub slot: SlotKind,
    pub parent_action_ids: Vec<ActionId>,
    pub clearing_price: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardEvent {
    pub action_id: ActionId,
    pub empirical_gain: f64,
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
