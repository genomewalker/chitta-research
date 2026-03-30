use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

pub use cr_types::VerifierSpec;
use cr_types::*;
use cr_graph::BeliefGraph;

#[derive(Debug, Serialize, Deserialize)]
pub struct AgendaConfig {
    pub programs: Vec<ProgramConfig>,
    pub budget: BudgetConfig,
    pub llm: LlmConfig,
    pub chitta: ChittaConfig,
    #[serde(default)]
    pub codebase_path: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProgramConfig {
    pub title: String,
    pub domain: String,
    pub questions: Vec<String>,
    #[serde(default)]
    pub methods: Vec<String>,
    #[serde(default = "default_priority")]
    pub priority: f32,
    #[serde(default = "default_budget")]
    pub max_budget_usd: f32,
    /// Optional external verifier. When present, Kriya uses it as a promotion
    /// gate instead of the built-in LOC/test heuristics.
    #[serde(default)]
    pub verifier: Option<VerifierSpec>,
}

fn default_priority() -> f32 { 1.0 }
fn default_budget() -> f32 { 5.0 }

#[derive(Debug, Serialize, Deserialize)]
pub struct BudgetConfig {
    pub total_usd: f32,
    #[serde(default)]
    pub gpu_slots: u32,
    #[serde(default = "default_cpu_workers")]
    pub cpu_workers: u32,
}

fn default_cpu_workers() -> u32 { 4 }

#[derive(Debug, Serialize, Deserialize)]
pub struct LlmConfig {
    pub provider: String,
    pub model: String,
    pub api_key_env: Option<String>,
    /// When `provider = "room"`, this block defines participants instead of
    /// using the built-in `standard_room` preset.
    #[serde(default)]
    pub room: Option<RoomConfig>,
}

/// Top-level room configuration embedded in the agenda YAML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomConfig {
    /// Number of debate rounds. Default: 2.
    #[serde(default = "default_room_rounds")]
    pub rounds: usize,
    pub participants: Vec<ParticipantConfig>,
    /// Synthesizer participant. If absent the last thread entry is returned verbatim.
    #[serde(default)]
    pub synthesizer: Option<ParticipantConfig>,
}

fn default_room_rounds() -> usize { 2 }

/// One participant in a configurable room.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipantConfig {
    /// Display name shown in the debate thread.
    pub name: String,
    /// Backend to use: `claude-cli`, `codex`, `anthropic`, `openai`, `gemini`,
    /// or `openai-compat` (for local models via Ollama / LM Studio).
    pub backend: String,
    /// Model name (required for `anthropic`, `openai`, `gemini`, `openai-compat`, `codex`).
    #[serde(default)]
    pub model: Option<String>,
    /// Environment variable holding the API key. Defaults per backend:
    /// anthropic→ANTHROPIC_API_KEY, openai→OPENAI_API_KEY, gemini→GOOGLE_API_KEY.
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Base URL override for `openai-compat` (e.g. `http://localhost:11434/v1`).
    #[serde(default)]
    pub base_url: Option<String>,
    /// System prompt for this participant's role.
    /// Defaults to a generic analytical role if absent.
    #[serde(default)]
    pub system: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChittaConfig {
    pub mind_path: String,
}

impl AgendaConfig {
    pub fn from_yaml(yaml: &str) -> Result<Self> {
        serde_yaml::from_str(yaml).context("failed to parse agenda YAML")
    }

    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        Self::from_yaml(&content)
    }

    pub fn into_belief_graph(self) -> Result<BeliefGraph> {
        let mut graph = BeliefGraph::new();

        for prog_cfg in &self.programs {
            let prog_id = NodeId::new();
            let prog_node = TypedNode::new(
                prog_id,
                NodeKind::ResearchProgram(ResearchProgram {
                    title: prog_cfg.title.clone(),
                    domain: prog_cfg.domain.clone(),
                    priority: prog_cfg.priority,
                    max_budget_usd: prog_cfg.max_budget_usd,
                }),
            );
            graph.add_node(prog_node).context("add program node")?;

            for q_text in &prog_cfg.questions {
                let q_id = NodeId::new();
                let q_node = TypedNode::new(
                    q_id,
                    NodeKind::Question(Question {
                        text: q_text.clone(),
                        program_id: prog_id,
                    }),
                );
                graph.add_node(q_node).context("add question node")?;
                graph
                    .add_edge(
                        q_id,
                        prog_id,
                        EpistemicEdge {
                            kind: EdgeKind::DerivedFrom,
                            weight: 1.0,
                            evidence_ids: vec![],
                        },
                    )
                    .context("add question->program edge")?;
            }

            for m_name in &prog_cfg.methods {
                let m_id = NodeId::new();
                let m_node = TypedNode::new(
                    m_id,
                    NodeKind::Method(Method {
                        name: m_name.clone(),
                        description: String::new(),
                    }),
                );
                graph.add_node(m_node).context("add method node")?;
                graph
                    .add_edge(
                        m_id,
                        prog_id,
                        EpistemicEdge {
                            kind: EdgeKind::Supports,
                            weight: 1.0,
                            evidence_ids: vec![],
                        },
                    )
                    .context("add method->program edge")?;
            }
        }

        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE_YAML: &str = r#"
programs:
  - title: "ARG prevalence in ancient permafrost"
    domain: metagenomics
    questions:
      - "Do antibiotic resistance genes increase with permafrost depth?"
      - "Are ancient ARGs phylogenetically distinct from modern variants?"
    methods:
      - "BLASTp against CARD"
      - "Short-read assembly with MEGAHIT"
    priority: 1.0
    max_budget_usd: 5.0
budget:
  total_usd: 10.0
  gpu_slots: 0
  cpu_workers: 4
llm:
  provider: anthropic
  model: claude-sonnet-4-6
  api_key_env: ANTHROPIC_API_KEY
chitta:
  mind_path: ~/.claude/mind
"#;

    #[test]
    fn parse_yaml() {
        let config = AgendaConfig::from_yaml(EXAMPLE_YAML).unwrap();
        assert_eq!(config.programs.len(), 1);
        assert_eq!(config.programs[0].questions.len(), 2);
        assert_eq!(config.programs[0].methods.len(), 2);
        assert_eq!(config.budget.total_usd, 10.0);
        assert_eq!(config.llm.model, "claude-sonnet-4-6");
    }

    #[test]
    fn into_belief_graph() {
        let config = AgendaConfig::from_yaml(EXAMPLE_YAML).unwrap();
        let graph = config.into_belief_graph().unwrap();
        assert_eq!(graph.node_count(), 5);
        assert_eq!(graph.edge_count(), 4);
    }

    #[test]
    fn minimal_yaml() {
        let yaml = r#"
programs:
  - title: "Test"
    domain: test
    questions: []
budget:
  total_usd: 1.0
llm:
  provider: anthropic
  model: claude-sonnet-4-6
  api_key_env: ANTHROPIC_API_KEY
chitta:
  mind_path: /tmp/mind
"#;
        let config = AgendaConfig::from_yaml(yaml).unwrap();
        assert_eq!(config.programs[0].priority, 1.0);
        assert_eq!(config.budget.cpu_workers, 4);
    }
}

#[cfg(test)]
mod nan_priority_tests {
    use super::*;

    #[test]
    fn nan_priority_yaml_parses_and_builds_graph() {
        let yaml = r#"
programs:
  - title: "NaN priority"
    domain: test
    questions: []
    priority: .nan
budget:
  total_usd: 1.0
llm:
  provider: anthropic
  model: claude-sonnet-4-6
  api_key_env: ANTHROPIC_API_KEY
chitta:
  mind_path: /tmp/mind
"#;
        let config = AgendaConfig::from_yaml(yaml).unwrap();
        assert!(config.programs[0].priority.is_nan(), "expected .nan to parse as NaN");
        let graph = config.into_belief_graph().unwrap();
        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.edge_count(), 0);
    }
}
