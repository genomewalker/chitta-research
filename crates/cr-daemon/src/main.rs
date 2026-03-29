use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use clap::Parser;
use tokio::sync::{mpsc, RwLock, Mutex};
use tracing::{error, info, warn};

use cr_agenda::AgendaConfig;
use cr_agents::{Agent, AgentContext, AgentEvent, ResearchAgenda};
use cr_agents::hotr::Hotr;
use cr_agents::adhvaryu::Adhvaryu;
use cr_agents::udgatr::Udgatr;
use cr_artifacts::ArtifactStore;
use cr_chitta::ChittaClient;

use cr_llm::{AnthropicClient, ClaudeCliClient, CodexClient, LlmClient, MockLlmClient, OpenAiClient, standard_room};
use cr_resources::ResourceManager;

#[derive(Parser)]
#[command(name = "chitta-research", about = "Autonomous research orchestrator")]
struct Cli {
    #[arg(long)]
    agenda: PathBuf,
    #[arg(long, default_value = "artifacts")]
    artifact_dir: PathBuf,
    #[arg(long, default_value = "graph_state.json")]
    graph_output: PathBuf,
    #[arg(long, default_value = "100")]
    max_cycles: u64,
    /// Use mock LLM — no external calls, for testing the full pipeline
    #[arg(long)]
    mock: bool,
}

fn build_llm_client(config: &cr_agenda::LlmConfig) -> Arc<dyn LlmClient> {
    match config.provider.as_str() {
        "claude-cli" => {
            info!("using claude -p (no API key needed)");
            Arc::new(ClaudeCliClient::with_model(config.model.clone()))
        }
        "anthropic" => {
            let api_key = std::env::var(config.api_key_env.as_deref().unwrap_or("ANTHROPIC_API_KEY"))
                .unwrap_or_else(|_| { warn!("ANTHROPIC_API_KEY not set"); "dummy-key".to_string() });
            Arc::new(AnthropicClient::new(api_key))
        }
        "openai" => {
            let api_key = std::env::var(config.api_key_env.as_deref().unwrap_or("OPENAI_API_KEY"))
                .unwrap_or_else(|_| { warn!("OPENAI_API_KEY not set"); "dummy-key".to_string() });
            Arc::new(OpenAiClient::openai(api_key))
        }
        "codex" => {
            info!("using codex exec (model: {})", config.model);
            Arc::new(CodexClient::with_model(config.model.clone()))
        }
        "room" => {
            // Multi-model discussion room: claude-cli (Critic) + codex (Empiricist)
            // debate each prompt for `rounds` rounds, then synthesize with claude-cli.
            info!("using discussion room (claude-cli + codex, 2 rounds)");
            Arc::new(standard_room(config.model.clone()).build())
        }
        other => {
            warn!("Unknown provider '{}', defaulting to claude-cli", other);
            Arc::new(ClaudeCliClient::new())
        }
    }
}

async fn agent_loop(
    agent: Box<dyn Agent>,
    ctx: Arc<AgentContext>,
    shutdown: Arc<AtomicBool>,
) {
    let name = agent.name().to_string();
    loop {
        if shutdown.load(Ordering::Relaxed) {
            info!(agent = %name, "shutting down");
            break;
        }
        if ctx.resources.budget_exhausted() {
            info!(agent = %name, "budget exhausted, stopping");
            break;
        }

        let start = std::time::Instant::now();
        match agent.step(&ctx).await {
            Ok(action) => {
                let elapsed = start.elapsed().as_millis() as u64;
                let summary = match &action {
                    cr_agents::AgentAction::AddNode { kind, .. } => format!("added {:?}", std::mem::discriminant(kind)),
                    cr_agents::AgentAction::UpdateNode { id, .. } => format!("updated {}", id),
                    cr_agents::AgentAction::RequestRun { plan_id } => format!("ran plan {}", plan_id),
                    cr_agents::AgentAction::ScoreFitness { node_id, .. } => format!("scored {}", node_id),
                    cr_agents::AgentAction::TriggerReconsolidation { node_id } => format!("reconsolidated {}", node_id),
                    cr_agents::AgentAction::Noop => "noop".to_string(),
                };
                let _ = ctx.event_tx.send(AgentEvent::ActionCompleted {
                    agent: name.clone(),
                    action_summary: summary,
                    elapsed_ms: elapsed,
                }).await;
            }
            Err(e) => {
                error!(agent = %name, error = %e, "step failed");
                let _ = ctx.event_tx.send(AgentEvent::Error {
                    agent: name.clone(),
                    error: e.to_string(),
                }).await;
            }
        }

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();

    info!(agenda = %cli.agenda.display(), "loading agenda");
    let config = AgendaConfig::from_file(&cli.agenda)?;

    let llm: Arc<dyn LlmClient> = if cli.mock {
        info!("using mock LLM (--mock flag)");
        Arc::new(MockLlmClient::new())
    } else {
        build_llm_client(&config.llm)
    };
    let mind_path = config.chitta.mind_path.clone();
    let gpu_slots = config.budget.gpu_slots as usize;
    let cpu_workers = config.budget.cpu_workers as usize;
    let total_budget = config.budget.total_usd;
    let agenda = ResearchAgenda {
        title: config.programs.first().map(|p| p.title.clone()).unwrap_or_default(),
        domain: config.programs.first().map(|p| p.domain.clone()).unwrap_or_default(),
        questions: config.programs.first().map(|p| p.questions.clone()).unwrap_or_default(),
        max_budget_usd: total_budget,
        max_cycles: cli.max_cycles,
    };

    let graph = config.into_belief_graph()?;
    let node_count = graph.node_count();
    info!(nodes = node_count, "belief graph initialized from agenda");

    let graph = Arc::new(RwLock::new(graph));

    let mut chitta = ChittaClient::for_mind(&mind_path);
    let chitta_connected = chitta.connect().await.is_ok();
    if chitta_connected {
        info!("connected to chittad");
    } else {
        warn!("could not connect to chittad — running without soul memory");
    }
    let chitta = Arc::new(Mutex::new(chitta));

    let artifacts = Arc::new(Mutex::new(ArtifactStore::open_or_init(&cli.artifact_dir)?));
    info!(path = %cli.artifact_dir.display(), "artifact store ready");

    let resources = Arc::new(ResourceManager::new(gpu_slots, cpu_workers, total_budget as f64));

    let (event_tx, mut event_rx) = mpsc::channel::<AgentEvent>(256);

    let ctx = Arc::new(AgentContext {
        graph: graph.clone(),
        llm,
        chitta,
        artifacts,
        resources: resources.clone(),
        event_tx: event_tx.clone(),
        agenda,
    });

    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_flag = shutdown.clone();
    ctrlc::set_handler(move || {
        info!("received SIGINT, initiating shutdown");
        shutdown_flag.store(true, Ordering::Relaxed);
    })?;

    let hotr_handle = tokio::spawn(agent_loop(Box::new(Hotr), ctx.clone(), shutdown.clone()));
    let adhvaryu_handle = tokio::spawn(agent_loop(Box::new(Adhvaryu), ctx.clone(), shutdown.clone()));
    let udgatr_handle = tokio::spawn(agent_loop(Box::new(Udgatr), ctx.clone(), shutdown.clone()));

    drop(event_tx);
    drop(ctx); // main must not hold a Sender — agents own the only clones

    let mut noop_counts: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    let mut action_count = 0u64;
    let max_cycles = cli.max_cycles;

    while let Some(event) = event_rx.recv().await {
        match event {
            AgentEvent::ActionCompleted { agent, action_summary, elapsed_ms } => {
                info!(agent = %agent, action = %action_summary, elapsed_ms, "action completed");
                action_count += 1;

                if action_summary == "noop" {
                    *noop_counts.entry(agent).or_insert(0) += 1;
                    let all_idle = noop_counts.len() >= 3
                        && noop_counts.values().all(|&c| c >= max_cycles);
                    if all_idle {
                        info!("all agents idle for {} cycles, shutting down", max_cycles);
                        shutdown.store(true, Ordering::Relaxed);
                    }
                } else {
                    noop_counts.insert(agent, 0);
                }
            }
            AgentEvent::Error { agent, error } => {
                error!(agent = %agent, error = %error, "agent error");
                // Errors count toward idle detection — an agent that only errors
                // is not making progress.
                *noop_counts.entry(agent).or_insert(0) += 1;
                let all_idle = noop_counts.len() >= 3
                    && noop_counts.values().all(|&c| c >= max_cycles);
                if all_idle {
                    info!("all agents idle/erroring for {} cycles, shutting down", max_cycles);
                    shutdown.store(true, Ordering::Relaxed);
                }
            }
            AgentEvent::CycleComplete { cycle } => {
                info!(cycle, "cycle complete");
            }
        }

        if resources.budget_exhausted() {
            info!("budget exhausted, shutting down");
            shutdown.store(true, Ordering::Relaxed);
        }
    }

    let _ = hotr_handle.await;
    let _ = adhvaryu_handle.await;
    let _ = udgatr_handle.await;

    // Save graph snapshot
    let graph_guard = graph.read().await;
    let json = graph_guard.snapshot_to_json()?;
    std::fs::write(&cli.graph_output, &json)?;
    info!(
        path = %cli.graph_output.display(),
        nodes = graph_guard.node_count(),
        edges = graph_guard.edge_count(),
        actions = action_count,
        "graph snapshot saved"
    );

    Ok(())
}
