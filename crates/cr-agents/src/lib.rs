pub mod hotr;
pub mod adhvaryu;
pub mod udgatr;
pub mod researcher;
pub mod scout;
pub mod kriya;
pub mod brahman;

use std::sync::Arc;

use async_trait::async_trait;
use cr_artifacts::ArtifactStore;
use cr_chitta::ChittaClient;
use cr_graph::BeliefGraph;
use cr_llm::LlmClient;
use cr_resources::ResourceManager;
use cr_types::{EpistemicEdge, FitnessVector, NodeId, NodeKind, VerifierSpec};

#[async_trait]
pub trait Agent: Send + Sync {
    fn name(&self) -> &str;
    async fn step(&self, ctx: &AgentContext) -> Result<AgentAction, anyhow::Error>;
}

pub struct AgentContext {
    pub graph: Arc<tokio::sync::RwLock<BeliefGraph>>,
    pub llm: Arc<dyn LlmClient>,
    pub chitta: Arc<tokio::sync::Mutex<ChittaClient>>,
    pub artifacts: Arc<tokio::sync::Mutex<ArtifactStore>>,
    pub resources: Arc<ResourceManager>,
    pub event_tx: tokio::sync::mpsc::Sender<AgentEvent>,
    pub agenda: ResearchAgenda,
    /// Node IDs of the ResearchProgram nodes belonging to this agenda run.
    pub active_program_ids: std::collections::HashSet<NodeId>,
    /// Root path of the codebase being researched (e.g. /path/to/mcaf).
    pub codebase_path: String,
}

#[derive(Debug, Clone)]
pub struct ResearchAgenda {
    pub title: String,
    pub domain: String,
    pub questions: Vec<String>,
    pub max_budget_usd: f32,
    pub max_cycles: u64,
    /// Optional external verifier spec threaded from the agenda YAML.
    /// When present, Kriya uses it as a promotion gate instead of
    /// the built-in LOC/test heuristics.
    pub verifier: Option<VerifierSpec>,
}

pub enum AgentAction {
    AddNode { kind: NodeKind, edges: Vec<(NodeId, EpistemicEdge)> },
    UpdateNode { id: NodeId, new_fitness: Option<FitnessVector>, new_posterior: Option<f32> },
    RequestRun { plan_id: NodeId },
    ScoreFitness { node_id: NodeId, fitness: FitnessVector },
    TriggerReconsolidation { node_id: NodeId },
    Noop,
}

pub enum AgentEvent {
    ActionCompleted { agent: String, action_summary: String, elapsed_ms: u64 },
    Error { agent: String, error: String },
    CycleComplete { cycle: u64 },
}
