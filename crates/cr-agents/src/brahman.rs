/// Brahman — the meta-controller. The mind that programs the research mind.
///
/// Brahman observes the full belief graph and chitta memory store, detects
/// stagnation and coverage gaps, and generates new research agendas — writing
/// agenda YAML files and signalling the daemon to restart on the new agenda.
///
/// For a life scientist, Brahman does three things:
/// 1. **Portfolio allocation**: which open questions deserve more cycles, which are saturated
/// 2. **Cross-domain transfer**: findings in one domain that imply hypotheses in another
/// 3. **Agenda generation**: writes new agenda.yaml files grounded in what the system
///    already knows — connecting chitta memories to new research questions
///
/// Brahman runs on a long timer (every 30+ minutes) and is intentionally conservative:
/// it only intervenes when it detects stagnation or a clear transfer opportunity.

use async_trait::async_trait;
use cr_llm::{CompletionRequest, Message};
use cr_types::*;
use std::collections::HashMap;

use crate::{Agent, AgentAction, AgentContext};

const SYSTEM_PROMPT: &str = r#"You are Brahman, the meta-controller of a scientific research system.

You observe the current state of a belief graph (hypotheses, claims, open questions) and
the researcher's soul memory (past knowledge, corrections, preferences, ongoing projects).

Your job:
1. Identify which research questions are SATURATED (many claims, little new) vs OPEN (few or no claims)
2. Detect TRANSFER opportunities: findings in one domain that imply testable hypotheses in another
3. Generate the NEXT research agenda — a concrete YAML agenda focused on the most valuable open territory

The researcher is a computational biologist / life scientist working on:
- Ancient metagenomics (aDNA from permafrost, sediments)
- Microbial ecology and diversity
- Metagenomic binning and assembly
- Damage patterns in ancient DNA
- Gene prediction in ancient samples
- Bioinformatics tool development (Rust, C++, Python, R)

The research system (chitta-research) is a Rust autonomous research loop.
Findings about chitta-research itself are also valuable — the system improves itself.

REQUIRED OUTPUT FORMAT — respond with ONLY this JSON:
{
  "assessment": "2-3 sentences on current graph state and portfolio balance",
  "stagnation": true or false,
  "stagnation_reason": "why if true",
  "transfer_opportunity": "one cross-domain insight, or null",
  "next_agenda": {
    "title": "concise title",
    "domain": "domain name",
    "questions": ["question1", "question2", "question3"],
    "methods": ["method1", "method2"],
    "priority": 1.0,
    "rationale": "why this is the highest-value next direction"
  }
}"#;

/// How often Brahman activates (in seconds)
const BRAHMAN_INTERVAL_SECS: u64 = 1800; // 30 minutes

/// Minimum novelty score on recent claims before triggering stagnation
const STAGNATION_NOVELTY_FLOOR: f32 = 0.40;

/// ── Research constitution ─────────────────────────────────────────────────
/// Rules that prevent infinite self-referential loops.

/// Max consecutive self-improvement agendas before Brahman refuses to generate another.
/// Forces a domain switch or hard stop rather than infinite meta-research.
const MAX_SELF_IMPROVEMENT_DEPTH: u32 = 3;

/// If novelty stays below this for 2 consecutive activations, hard stop — the
/// graph is fully saturated and generating more agendas won't help.
const HARD_STOP_NOVELTY: f32 = 0.20;

/// Don't generate a new agenda if more than this many hypotheses are untested.
/// Generation is outpacing execution — let Adhvaryu/Udgatr catch up first.
const MAX_HYPOTHESIS_BACKLOG: usize = 10;

pub struct Brahman {
    last_activation: std::sync::Mutex<std::time::Instant>,
    /// Count of consecutive self-improvement agendas generated this session
    self_improvement_depth: std::sync::atomic::AtomicU32,
    /// Novelty score from previous activation (for hard-stop detection)
    prev_novelty: std::sync::Mutex<f32>,
}

impl Brahman {
    pub fn new() -> Self {
        Self {
            // First activation after 5 minutes — let other agents build context first
            last_activation: std::sync::Mutex::new(
                std::time::Instant::now() - std::time::Duration::from_secs(BRAHMAN_INTERVAL_SECS - 300)
            ),
            self_improvement_depth: std::sync::atomic::AtomicU32::new(0),
            prev_novelty: std::sync::Mutex::new(1.0),
        }
    }
}

fn question_claim_counts(nodes: &[&TypedNode]) -> HashMap<NodeId, usize> {
    // Count how many claims exist per research question (via hypothesis → claim chain)
    // For simplicity: count claims per program for now
    let mut counts: HashMap<NodeId, usize> = HashMap::new();
    for n in nodes {
        if let NodeKind::Claim(_) = &n.kind {
            *counts.entry(NodeId::new()).or_insert(0) += 1; // placeholder
        }
    }
    counts
}

fn recent_novelty(nodes: &[&TypedNode]) -> f32 {
    let runs: Vec<_> = nodes.iter()
        .filter(|n| matches!(n.kind, NodeKind::Run(_)))
        .filter(|n| n.fitness.is_some())
        .collect();
    if runs.is_empty() { return 1.0; }
    // Average novelty of the last 5 runs
    let recent: Vec<f32> = runs.iter().rev().take(5)
        .filter_map(|n| n.fitness.map(|f| f.novelty))
        .collect();
    if recent.is_empty() { return 1.0; }
    recent.iter().sum::<f32>() / recent.len() as f32
}

fn build_graph_summary(nodes: &[&TypedNode]) -> String {
    let programs: Vec<_> = nodes.iter().filter(|n| matches!(n.kind, NodeKind::ResearchProgram(_))).collect();
    let questions: Vec<_> = nodes.iter().filter(|n| matches!(n.kind, NodeKind::Question(_))).collect();
    let hyps: Vec<_> = nodes.iter().filter(|n| matches!(n.kind, NodeKind::Hypothesis(_))).collect();
    let runs: Vec<_> = nodes.iter().filter(|n| matches!(n.kind, NodeKind::Run(_))).collect();
    let claims: Vec<_> = nodes.iter().filter(|n| matches!(n.kind, NodeKind::Claim(_))).collect();

    let tested = hyps.iter().filter(|h| {
        if let NodeKind::Hypothesis(hh) = &h.kind { hh.posterior_confidence.is_some() } else { false }
    }).count();

    let ok_runs = runs.iter().filter(|r| {
        if let NodeKind::Run(rr) = &r.kind { rr.status == RunStatus::Succeeded } else { false }
    }).count();

    let top_claims: Vec<String> = {
        let mut c: Vec<_> = claims.iter().filter_map(|n| {
            if let NodeKind::Claim(c) = &n.kind { Some((c.confidence, c.statement.clone())) } else { None }
        }).collect();
        c.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        c.into_iter().take(5).map(|(conf, s)| format!("[{conf:.2}] {}", &s[..s.len().min(100)])).collect()
    };

    let open_qs: Vec<String> = questions.iter().filter_map(|n| {
        if let NodeKind::Question(q) = &n.kind { Some(q.text.clone()) } else { None }
    }).take(5).collect();

    format!(
        "Programs: {}\nQuestions: {} ({} open)\nHypotheses: {} ({} tested)\nRuns: {} ({} succeeded)\nClaims: {}\n\nTop claims:\n{}\n\nOpen questions:\n{}",
        programs.len(),
        questions.len(), questions.len(),
        hyps.len(), tested,
        runs.len(), ok_runs,
        claims.len(),
        top_claims.join("\n"),
        open_qs.join("\n"),
    )
}

#[async_trait]
impl Agent for Brahman {
    fn name(&self) -> &str { "brahman" }

    async fn step(&self, ctx: &AgentContext) -> Result<AgentAction, anyhow::Error> {
        // Throttle — only run every BRAHMAN_INTERVAL_SECS
        {
            let last = self.last_activation.lock().unwrap();
            if last.elapsed().as_secs() < BRAHMAN_INTERVAL_SECS {
                return Ok(AgentAction::Noop);
            }
        }
        *self.last_activation.lock().unwrap() = std::time::Instant::now();

        let (graph_summary, novelty, untested_hyps) = {
            let graph = ctx.graph.read().await;
            let nodes: Vec<&TypedNode> = graph.all_nodes().into_iter().collect();
            let summary = build_graph_summary(&nodes);
            let novelty = recent_novelty(&nodes);
            let untested = nodes.iter().filter(|n| {
                if let NodeKind::Hypothesis(h) = &n.kind { h.posterior_confidence.is_none() } else { false }
            }).count();
            (summary, novelty, untested)
        };

        // ── Research constitution checks ──────────────────────────────────
        // Rule 1: Hypothesis backlog gate — let Adhvaryu/Udgatr catch up first
        if untested_hyps > MAX_HYPOTHESIS_BACKLOG {
            tracing::info!(
                untested = untested_hyps,
                limit = MAX_HYPOTHESIS_BACKLOG,
                "brahman: backlog gate — waiting for Adhvaryu to clear hypothesis queue"
            );
            return Ok(AgentAction::Noop);
        }

        // Rule 2: Hard novelty stop — graph is fully saturated
        let prev = *self.prev_novelty.lock().unwrap();
        *self.prev_novelty.lock().unwrap() = novelty;
        if novelty < HARD_STOP_NOVELTY && prev < HARD_STOP_NOVELTY {
            tracing::warn!(
                novelty,
                prev_novelty = prev,
                "brahman: HARD STOP — novelty below floor for 2 consecutive activations, \
                 graph is saturated. Run cr-report and start a new agenda manually."
            );
            if let Ok(mut chitta) = ctx.chitta.try_lock() {
                let _ = chitta.remember(
                    &format!("[brahman:hard-stop] Novelty {novelty:.2} for 2 activations — \
                              graph saturated. Manual intervention required."),
                    "wisdom", &["brahman", "hard-stop"], 0.95,
                ).await;
            }
            return Ok(AgentAction::Noop);
        }

        // Rule 3: Self-improvement depth cap
        let depth = self.self_improvement_depth.load(std::sync::atomic::Ordering::Relaxed);
        if depth >= MAX_SELF_IMPROVEMENT_DEPTH {
            tracing::warn!(
                depth,
                limit = MAX_SELF_IMPROVEMENT_DEPTH,
                "brahman: self-improvement depth cap reached — refusing to generate \
                 another self-referential agenda. Run cr-report and switch domains."
            );
            if let Ok(mut chitta) = ctx.chitta.try_lock() {
                let _ = chitta.remember(
                    &format!("[brahman:depth-cap] {depth} consecutive self-improvement agendas — \
                              refusing to generate more. Switch to a domain agenda."),
                    "wisdom", &["brahman", "depth-cap"], 0.95,
                ).await;
            }
            return Ok(AgentAction::Noop);
        }

        // Recall relevant memories from chitta for context
        let chitta_context = {
            let mut chitta = ctx.chitta.lock().await;
            if chitta.connect().await.is_ok() {
                let hits = chitta.recall(
                    &format!("{} research gaps opportunities", ctx.agenda.domain),
                    8,
                ).await.unwrap_or_default();
                hits.into_iter()
                    .map(|h| format!("• {}", &h.content[..h.content.len().min(150)]))
                    .collect::<Vec<_>>()
                    .join("\n")
            } else {
                String::new()
            }
        };

        let stagnation_signal = novelty < STAGNATION_NOVELTY_FLOOR;

        let user_msg = format!(
            "CURRENT BELIEF GRAPH STATE:\n{graph_summary}\n\n\
             RECENT NOVELTY SCORE: {novelty:.2} (stagnation threshold: {STAGNATION_NOVELTY_FLOOR})\n\n\
             RELEVANT MEMORIES FROM CHITTA:\n{chitta_context}\n\n\
             RESEARCHER DOMAIN: {domain}\n\n\
             Generate your assessment and next agenda.\n\n\
             REQUIRED OUTPUT FORMAT — ONLY JSON:\n\
             {{\"assessment\":\"...\",\"stagnation\":true/false,\"stagnation_reason\":\"...\",\
             \"transfer_opportunity\":\"...\",\"next_agenda\":{{\"title\":\"...\",\"domain\":\"...\",\
             \"questions\":[\"...\"],\"methods\":[\"...\"],\"priority\":1.0,\"rationale\":\"...\"}}}}",
            domain = ctx.agenda.domain,
        );

        let resp = ctx.llm.complete(CompletionRequest {
            model: String::new(),
            system: SYSTEM_PROMPT.to_string(),
            messages: vec![Message { role: "user".into(), content: user_msg }],
            max_tokens: 1500,
            temperature: 0.4,
        }).await?;

        let content = resp.content.trim();
        let json_str = if let Some(s) = content.find('{') {
            if let Some(e) = content.rfind('}') { &content[s..=e] } else { content }
        } else { content };

        let assessment: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| anyhow::anyhow!("Brahman parse failed: {e}"))?;

        let assess_text = assessment["assessment"].as_str().unwrap_or("");
        let is_stagnant = assessment["stagnation"].as_bool().unwrap_or(false);
        let transfer   = assessment["transfer_opportunity"].as_str().unwrap_or("").to_string();
        let next       = &assessment["next_agenda"];

        tracing::info!(
            assessment = %assess_text,
            stagnation = is_stagnant,
            novelty,
            "brahman: portfolio assessment"
        );

        // Store assessment in chitta
        if let Ok(mut chitta) = ctx.chitta.try_lock() {
            let _ = chitta.remember(
                &format!("[brahman:assessment] {assess_text}"),
                "wisdom",
                &["brahman", "portfolio", "chitta-research"],
                0.85,
            ).await;

            if !transfer.is_empty() && transfer != "null" {
                let _ = chitta.remember(
                    &format!("[brahman:transfer] {transfer}"),
                    "wisdom",
                    &["brahman", "transfer", "cross-domain"],
                    0.80,
                ).await;
                tracing::info!(transfer = %transfer, "brahman: cross-domain transfer opportunity stored");
            }
        }

        // Write next agenda if stagnation detected or significant transfer opportunity found
        if is_stagnant || (stagnation_signal && !transfer.is_empty()) {
            let agenda_title = next["title"].as_str().unwrap_or("brahman-generated");
            let agenda_domain = next["domain"].as_str().unwrap_or(&ctx.agenda.domain);
            let rationale = next["rationale"].as_str().unwrap_or("");
            let questions: Vec<String> = next["questions"].as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();
            let methods: Vec<String> = next["methods"].as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();

            let questions_yaml = questions.iter().map(|q| format!("      - \"{}\"", q.replace('"', "'"))).collect::<Vec<_>>().join("\n");
            let methods_yaml   = methods.iter().map(|m| format!("      - \"{}\"", m.replace('"', "'"))).collect::<Vec<_>>().join("\n");

            let agenda_yaml = format!(r#"# Generated by Brahman
# Rationale: {rationale}
programs:
  - title: "{agenda_title}"
    domain: {agenda_domain}
    questions:
{questions_yaml}
    methods:
{methods_yaml}
    priority: 1.0
    max_budget_usd: 10.0

budget:
  total_usd: 10.0
  gpu_slots: 0
  cpu_workers: 4

llm:
  provider: room
  model: "{agenda_title}"

chitta:
  mind_path: ~/.claude/mind
"#);

            let agenda_path = format!("agenda.brahman-{}.yaml",
                chrono::Utc::now().format("%Y%m%d-%H%M%S"));
            std::fs::write(&agenda_path, &agenda_yaml)?;

            // Track self-improvement depth for the constitution cap
            let is_self_improvement = agenda_domain == "software_engineering"
                || agenda_title.to_lowercase().contains("chitta")
                || agenda_title.to_lowercase().contains("cresearch")
                || agenda_title.to_lowercase().contains("test")
                || agenda_title.to_lowercase().contains("fix");
            if is_self_improvement {
                self.self_improvement_depth.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            } else {
                self.self_improvement_depth.store(0, std::sync::atomic::Ordering::Relaxed);
            }

            tracing::info!(
                path = %agenda_path,
                title = %agenda_title,
                domain = %agenda_domain,
                self_improvement_depth = self.self_improvement_depth.load(std::sync::atomic::Ordering::Relaxed),
                "brahman: wrote new agenda (stagnation detected)"
            );

            // Store the new agenda path in chitta so the researcher knows about it
            if let Ok(mut chitta) = ctx.chitta.try_lock() {
                let _ = chitta.remember(
                    &format!("[brahman:new-agenda] {agenda_path}\nTitle: {agenda_title}\nRationale: {rationale}"),
                    "milestone",
                    &["brahman", "agenda", "next-direction"],
                    0.90,
                ).await;
            }
        }

        Ok(AgentAction::Noop)
    }
}
