/// cr-report — research journal HTML report from graph_state.json
///
/// Usage:
///   cr-report --graph graph_state.json --artifacts ./artifacts --output report.html
///   cr-report   # uses defaults in current directory

use anyhow::{Context, Result};
use clap::Parser;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "cr-report", about = "Generate research journal HTML from chitta-research graph")]
struct Cli {
    #[arg(long, default_value = "graph_state.json")]
    graph: PathBuf,
    #[arg(long, default_value = "artifacts")]
    artifacts: PathBuf,
    #[arg(long, default_value = "report.html")]
    output: PathBuf,
}

// ── Domain types ─────────────────────────────────────────────────────────────

/// A node kind tag + its payload, extracted from the JSON envelope.
struct TypedNode<'a> {
    id: &'a str,
    kind: &'a str,
    data: &'a Value,
    created_at: &'a str,
    fitness: Option<&'a Value>,
}

/// Artifact data loaded from artifacts/{run-dir}/results.json
struct ArtifactData {
    outcome: String,
    summary: String,
    observations: Vec<String>,
    plan_steps: Vec<String>,
    metrics: Option<Value>,
}

/// Debate turn loaded from artifacts/{run-dir}/debate.json
struct DebateTurn {
    participant: String,
    message: String,
}

// ── Graph accessors ───────────────────────────────────────────────────────────

fn typed(n: &Value) -> Option<TypedNode<'_>> {
    let kind_obj = n["kind"].as_object()?;
    let (kind, data) = kind_obj.iter().next()?;
    Some(TypedNode {
        id: n["id"].as_str().unwrap_or(""),
        kind: kind.as_str(),
        data,
        created_at: n["created_at"].as_str().unwrap_or(""),
        fitness: if n["fitness"].is_null() { None } else { Some(&n["fitness"]) },
    })
}

fn s(v: &Value, key: &str) -> String {
    v[key].as_str().unwrap_or("").to_string()
}

fn esc(raw: &str) -> String {
    raw.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn fmt_timestamp(ts: &str) -> String {
    // "2026-03-29T07:17:17.592327100Z" -> "2026-03-29 07:17 UTC"
    if ts.len() >= 16 {
        format!("{} {} UTC", &ts[..10], &ts[11..16])
    } else {
        ts.to_string()
    }
}

fn confidence_pct(c: f64) -> u64 {
    (c * 100.0).round() as u64
}

fn confidence_color(c: f64) -> &'static str {
    if c >= 0.7 { "#2d6a4f" } else if c >= 0.45 { "#8a5c00" } else { "#b94040" }
}

fn confidence_label(c: f64) -> &'static str {
    if c >= 0.7 { "high" } else if c >= 0.45 { "medium" } else { "low" }
}

// ── Artifact loading ──────────────────────────────────────────────────────────

/// Build a run-dir name from the run's started_at timestamp.
/// "2026-03-29T07:17:17.592327100Z" -> "run-20260329-071717"
fn run_dir_name(started_at: &str) -> String {
    if started_at.len() < 19 {
        return String::new();
    }
    let date = started_at[..10].replace('-', "");
    let time = started_at[11..19].replace(':', "");
    format!("run-{date}-{time}")
}

fn load_artifact(artifacts_dir: &PathBuf, dir_name: &str) -> Option<ArtifactData> {
    let path = artifacts_dir.join(dir_name).join("results.json");
    let text = fs::read_to_string(&path).ok()?;
    let v: Value = serde_json::from_str(&text).ok()?;
    Some(ArtifactData {
        outcome: s(&v, "outcome"),
        summary: s(&v, "summary"),
        observations: v["observations"]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect())
            .unwrap_or_default(),
        plan_steps: v["plan_steps"]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect())
            .unwrap_or_default(),
        metrics: if v["metrics"].is_object() { Some(v["metrics"].clone()) } else { None },
    })
}

fn load_debate(artifacts_dir: &PathBuf, dir_name: &str) -> Vec<DebateTurn> {
    let path = artifacts_dir.join(dir_name).join("debate.json");
    let text = match fs::read_to_string(&path) {
        Ok(t) => t,
        Err(_) => return vec![],
    };
    let v: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => return vec![],
    };
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|turn| {
                    Some(DebateTurn {
                        participant: s(turn, "participant"),
                        message: s(turn, "message"),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

// ── Edge traversal ────────────────────────────────────────────────────────────

/// out_edges[from_id] = Vec<(to_id, edge_kind)>
/// in_edges[to_id]   = Vec<(from_id, edge_kind)>
fn build_edge_maps(graph: &Value) -> (HashMap<String, Vec<(String, String)>>, HashMap<String, Vec<(String, String)>>) {
    let mut out: HashMap<String, Vec<(String, String)>> = HashMap::new();
    let mut inn: HashMap<String, Vec<(String, String)>> = HashMap::new();
    if let Some(edges) = graph["edges"].as_array() {
        for e in edges {
            let from = e["from"].as_str().unwrap_or("").to_string();
            let to = e["to"].as_str().unwrap_or("").to_string();
            let kind = e["edge"]["kind"].as_str().unwrap_or("").to_string();
            out.entry(from.clone()).or_default().push((to.clone(), kind.clone()));
            inn.entry(to).or_default().push((from, kind));
        }
    }
    (out, inn)
}

fn parents_of_kind<'a>(
    id: &str,
    edge_kind: &str,
    node_kind: &str,
    in_edges: &HashMap<String, Vec<(String, String)>>,
    node_map: &'a HashMap<&str, TypedNode<'a>>,
) -> Vec<&'a TypedNode<'a>> {
    in_edges
        .get(id)
        .map(|edges| {
            edges
                .iter()
                .filter(|(_, ek)| ek == edge_kind)
                .filter_map(|(fid, _)| node_map.get(fid.as_str()))
                .filter(|n| n.kind == node_kind)
                .collect()
        })
        .unwrap_or_default()
}

// ── HTML rendering helpers ────────────────────────────────────────────────────

fn render_confidence_bar(conf: f64, width_px: u32) -> String {
    let pct = confidence_pct(conf);
    let color = confidence_color(conf);
    let label = confidence_label(conf);
    format!(
        r#"<div class="conf-meter">
          <div class="conf-track" style="width:{width_px}px">
            <div class="conf-fill" style="width:{pct}%;background:{color}"></div>
          </div>
          <span class="conf-value" style="color:{color}">{pct}% <em class="conf-label">{label}</em></span>
        </div>"#
    )
}

fn render_fitness(fitness: &Value) -> String {
    let keys = ["novelty", "empirical_gain", "reproducibility", "cost_efficiency", "transfer_potential", "calibration_improvement"];
    let mut cells = String::new();
    for key in &keys {
        let val = fitness[key].as_f64().unwrap_or(0.0);
        let pct = confidence_pct(val);
        let color = confidence_color(val);
        let label = key.replace('_', " ");
        cells.push_str(&format!(
            r#"<div class="fitness-cell">
              <div class="fitness-bar-wrap"><div class="fitness-bar" style="height:{pct}%;background:{color}"></div></div>
              <div class="fitness-name">{label}</div>
              <div class="fitness-val" style="color:{color}">{pct}</div>
            </div>"#
        ));
    }
    format!(r#"<div class="fitness-chart">{cells}</div>"#)
}

fn render_debate_block(turns: &[DebateTurn]) -> String {
    if turns.is_empty() {
        return r#"<div class="debate-absent">No debate recorded for this run.</div>"#.to_string();
    }
    let mut html = String::from(r#"<div class="debate-block"><div class="debate-heading">Debate Record</div>"#);
    for turn in turns {
        let (cls, role_label) = match turn.participant.to_lowercase().as_str() {
            p if p.contains("critic") => ("debate-critic", "Critic"),
            p if p.contains("empiricist") => ("debate-empiricist", "Empiricist"),
            p if p.contains("synth") => ("debate-synthesizer", "Synthesizer"),
            _ => ("debate-other", turn.participant.as_str()),
        };
        html.push_str(&format!(
            r#"<div class="debate-turn {cls}">
              <div class="debate-speaker">{}</div>
              <div class="debate-text">{}</div>
            </div>"#,
            esc(role_label),
            esc(&turn.message)
        ));
    }
    html.push_str("</div>");
    html
}

fn render_observations(observations: &[String]) -> String {
    if observations.is_empty() {
        return String::new();
    }
    let mut html = String::from(r#"<div class="observations"><div class="obs-heading">Observations</div>"#);
    for obs in observations {
        if obs.len() > 200 || obs.contains('\n') {
            let short = obs.chars().take(120).collect::<String>();
            html.push_str(&format!(
                r#"<details class="obs-item obs-long"><summary>{}</summary><div class="obs-full">{}</div></details>"#,
                esc(&short),
                esc(obs)
            ));
        } else {
            html.push_str(&format!(r#"<div class="obs-item">{}</div>"#, esc(obs)));
        }
    }
    html.push_str("</div>");
    html
}

fn render_plan_steps(steps: &[String]) -> String {
    if steps.is_empty() {
        return String::new();
    }
    let items: String = steps.iter()
        .map(|s| format!(r#"<li class="step-item">{}</li>"#, esc(s)))
        .collect();
    format!(r#"<ol class="plan-steps">{items}</ol>"#)
}

fn render_metrics(metrics: &Value) -> String {
    let obj = match metrics.as_object() {
        Some(o) => o,
        None => return String::new(),
    };
    let mut rows = String::new();
    for (k, v) in obj {
        let val_str = if let Some(f) = v.as_f64() {
            format!("{f:.4}")
        } else {
            v.to_string()
        };
        rows.push_str(&format!(
            r#"<tr><td class="metric-key">{}</td><td class="metric-val">{}</td></tr>"#,
            esc(k), esc(&val_str)
        ));
    }
    format!(r#"<details class="metrics-block"><summary>Metrics ({} values)</summary><table class="metrics-table">{rows}</table></details>"#, obj.len())
}

fn run_verdict_class(outcome: &str, status: &str) -> &'static str {
    match (outcome, status) {
        ("succeeded", _) | (_, "Succeeded") => "verdict-confirmed",
        _ => "verdict-refuted",
    }
}

fn run_verdict_label(outcome: &str, status: &str) -> &'static str {
    match (outcome, status) {
        ("succeeded", _) | (_, "Succeeded") => "Confirmed",
        _ => "Refuted",
    }
}

// ── Section renderers ─────────────────────────────────────────────────────────

fn render_hypothesis_entry(
    hyp: &TypedNode<'_>,
    question_text: &str,
    _out_edges: &HashMap<String, Vec<(String, String)>>,
    in_edges: &HashMap<String, Vec<(String, String)>>,
    node_map: &HashMap<&str, TypedNode<'_>>,
    artifacts_dir: &PathBuf,
) -> String {
    let hd = hyp.data;
    let statement = s(hd, "statement");
    let prior = hd["prior_confidence"].as_f64().unwrap_or(0.0);
    let posterior = hd["posterior_confidence"].as_f64();
    let tier = hd["tier"].as_u64().unwrap_or(0);
    let created = fmt_timestamp(hyp.created_at);

    // Find plan DerivedFrom this hypothesis (plan -> hypothesis)
    let plans = parents_of_kind(hyp.id, "DerivedFrom", "ExperimentPlan", in_edges, node_map);

    // Claims that Refute this hypothesis
    let refuting_claims = parents_of_kind(hyp.id, "Refutes", "Claim", in_edges, node_map);

    let has_posterior = posterior.is_some();
    let entry_class = if has_posterior { "journal-entry" } else { "journal-entry journal-pending" };

    let mut html = format!(
        r#"<div class="{entry_class}">
          <div class="entry-gutter">
            <div class="entry-tier">T{tier}</div>
            <div class="entry-date">{created}</div>
          </div>
          <div class="entry-body">"#
    );

    // Research question provenance
    if !question_text.is_empty() {
        html.push_str(&format!(
            r#"<div class="entry-question"><span class="question-mark">?</span> {}</div>"#,
            esc(question_text)
        ));
    }

    // Hypothesis statement + prior
    html.push_str(&format!(
        r#"<div class="entry-hypothesis">
          <div class="hyp-statement">{}</div>
          <div class="hyp-priors">
            <div class="prior-block">
              <span class="prior-label">Prior</span>
              {}
            </div>"#,
        esc(&statement),
        render_confidence_bar(prior, 140)
    ));

    if let Some(post) = posterior {
        let delta = post - prior;
        let delta_class = if delta >= 0.0 { "delta-up" } else { "delta-down" };
        let delta_sign = if delta >= 0.0 { "+" } else { "" };
        html.push_str(&format!(
            r#"<div class="posterior-block">
              <span class="prior-label">Posterior</span>
              {}
              <span class="{delta_class}">{delta_sign}{:.2} shift</span>
            </div>"#,
            render_confidence_bar(post, 140),
            delta
        ));
    }

    html.push_str("</div></div>"); // close hyp-priors, entry-hypothesis

    // For each plan, render the experiment + run
    for plan in &plans {
        let pd = plan.data;
        let description = s(pd, "description");

        // Find run DerivedFrom this plan (run -> plan)
        let runs = parents_of_kind(plan.id, "DerivedFrom", "Run", in_edges, node_map);

        html.push_str(r#"<div class="experiment-block">"#);
        html.push_str(&format!(r#"<div class="exp-label">Experiment</div>"#));
        if !description.is_empty() {
            html.push_str(&format!(r#"<div class="exp-description">{}</div>"#, esc(&description)));
        }

        // Plan steps — prefer artifact's plan_steps if available (they're the actually-executed ones)
        for run in &runs {
            let rd = run.data;
            let status = s(rd, "status");
            let started = fmt_timestamp(run.data["started_at"].as_str().unwrap_or(""));
            let dir_name = run_dir_name(run.data["started_at"].as_str().unwrap_or(""));

            let artifact = if !dir_name.is_empty() {
                load_artifact(artifacts_dir, &dir_name)
            } else {
                None
            };
            let debate = if !dir_name.is_empty() {
                load_debate(artifacts_dir, &dir_name)
            } else {
                vec![]
            };

            let outcome = artifact.as_ref().map(|a| a.outcome.as_str()).unwrap_or("unknown");
            let verdict_class = run_verdict_class(outcome, &status);
            let verdict_label = run_verdict_label(outcome, &status);

            html.push_str(&format!(
                r#"<div class="run-block {verdict_class}">
                  <div class="run-header">
                    <div class="run-verdict-badge {verdict_class}">{verdict_label}</div>
                    <div class="run-meta">{started}</div>
                  </div>"#
            ));

            // Plan steps (from artifact if available, else from plan node)
            let steps = artifact.as_ref()
                .filter(|a| !a.plan_steps.is_empty())
                .map(|a| a.plan_steps.as_slice())
                .unwrap_or_else(|| {
                    // fallback: steps from the plan node, but we can't store the vec here
                    // so return empty; we'll render from plan node separately
                    &[]
                });

            if !steps.is_empty() {
                html.push_str(&format!(r#"<div class="steps-label">Steps executed</div>"#));
                html.push_str(&render_plan_steps(steps));
            } else {
                // render from plan node
                let plan_steps: Vec<String> = pd["steps"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                    .unwrap_or_default();
                if !plan_steps.is_empty() {
                    html.push_str(&format!(r#"<div class="steps-label">Planned steps</div>"#));
                    html.push_str(&render_plan_steps(&plan_steps));
                }
            }

            // Debate
            html.push_str(&render_debate_block(&debate));

            // Observations + summary
            if let Some(art) = &artifact {
                html.push_str(&render_observations(&art.observations));

                if !art.summary.is_empty() {
                    html.push_str(&format!(
                        r#"<div class="run-summary"><div class="summary-label">Summary</div><div class="summary-text">{}</div></div>"#,
                        esc(&art.summary)
                    ));
                }

                if let Some(metrics) = &art.metrics {
                    html.push_str(&render_metrics(metrics));
                }
            }

            // Fitness breakdown
            if let Some(fitness) = run.fitness {
                html.push_str(&format!(r#"<div class="fitness-label">Fitness breakdown</div>"#));
                html.push_str(&render_fitness(fitness));
            }

            html.push_str("</div>"); // run-block
        }

        html.push_str("</div>"); // experiment-block
    }

    // Claims
    if !refuting_claims.is_empty() {
        html.push_str(r#"<div class="entry-claims">"#);
        for claim in &refuting_claims {
            let cd = claim.data;
            let stmt = s(cd, "statement");
            let conf = cd["confidence"].as_f64().unwrap_or(0.0);
            let color = confidence_color(conf);
            html.push_str(&format!(
                r#"<div class="claim-finding">
                  <div class="claim-finding-text">{}</div>
                  <div class="claim-conf-row">{}</div>
                </div>"#,
                esc(&stmt),
                render_confidence_bar(conf, 120)
            ));
            let _ = color;
        }
        html.push_str("</div>");
    }

    html.push_str("</div></div>"); // entry-body, journal-entry
    html
}

// ── Main render ───────────────────────────────────────────────────────────────

fn render(
    graph: &Value,
    out_edges: &HashMap<String, Vec<(String, String)>>,
    in_edges: &HashMap<String, Vec<(String, String)>>,
    node_map: &HashMap<&str, TypedNode<'_>>,
    artifacts_dir: &PathBuf,
) -> String {
    let nodes: Vec<TypedNode<'_>> = graph["nodes"]
        .as_array()
        .map(|a| a.iter().filter_map(typed).collect())
        .unwrap_or_default();

    // Partition by kind
    let programs:    Vec<&TypedNode<'_>> = nodes.iter().filter(|n| n.kind == "ResearchProgram").collect();
    let questions:   Vec<&TypedNode<'_>> = nodes.iter().filter(|n| n.kind == "Question").collect();
    let hypotheses:  Vec<&TypedNode<'_>> = nodes.iter().filter(|n| n.kind == "Hypothesis").collect();
    let runs:        Vec<&TypedNode<'_>> = nodes.iter().filter(|n| n.kind == "Run").collect();
    let all_claims:  Vec<&TypedNode<'_>> = nodes.iter().filter(|n| n.kind == "Claim").collect();

    let n_succeeded = runs.iter().filter(|r| s(r.data, "status") == "Succeeded").count();
    let n_failed    = runs.iter().filter(|r| s(r.data, "status") == "Failed").count();
    let n_with_post = hypotheses.iter().filter(|h| h.data["posterior_confidence"].as_f64().is_some()).count();

    // Map question id -> question text
    let question_text: HashMap<&str, &str> = questions.iter()
        .map(|q| (q.id, q.data["text"].as_str().unwrap_or("")))
        .collect();

    // Map hypothesis id -> parent question text
    // Hypothesis --DerivedFrom--> Question
    let hyp_to_question: HashMap<&str, &str> = hypotheses.iter()
        .map(|h| {
            let q_text = out_edges.get(h.id)
                .and_then(|edges| {
                    edges.iter()
                        .find(|(_, ek)| ek == "DerivedFrom")
                        .and_then(|(qid, _)| question_text.get(qid.as_str()).copied())
                })
                .unwrap_or("");
            (h.id, q_text)
        })
        .collect();

    // Group hypotheses by their parent question
    let mut question_to_hyps: HashMap<&str, Vec<&TypedNode<'_>>> = HashMap::new();
    let mut ungrouped_hyps: Vec<&TypedNode<'_>> = vec![];
    for h in &hypotheses {
        let q_text = hyp_to_question.get(h.id).copied().unwrap_or("");
        if q_text.is_empty() {
            ungrouped_hyps.push(h);
        } else {
            question_to_hyps.entry(q_text).or_default().push(h);
        }
    }

    // Order questions by their appearance in graph
    let ordered_questions: Vec<&str> = questions.iter()
        .map(|q| q.data["text"].as_str().unwrap_or(""))
        .filter(|t| !t.is_empty() && question_to_hyps.contains_key(t))
        .collect();

    // Confirmed claims (high confidence) vs all
    let confirmed_claims: Vec<&TypedNode<'_>> = all_claims.iter()
        .copied()
        .filter(|c| c.data["confidence"].as_f64().unwrap_or(0.0) >= 0.5)
        .collect();

    // Reverted/failed runs (Failed status)
    let failed_runs: Vec<&TypedNode<'_>> = runs.iter()
        .copied()
        .filter(|r| s(r.data, "status") == "Failed")
        .collect();

    // Build the page body
    let mut body = String::with_capacity(256 * 1024);

    // ── Header ──
    let program_title = programs.first()
        .map(|p| s(p.data, "title"))
        .unwrap_or_else(|| "Research Program".to_string());
    let program_domain = programs.first()
        .map(|p| s(p.data, "domain"))
        .unwrap_or_default();
    let run_ts = runs.iter()
        .filter_map(|r| r.data["started_at"].as_str())
        .min()
        .map(fmt_timestamp)
        .unwrap_or_else(|| "—".to_string());

    body.push_str(&format!(
        r#"<header class="journal-header">
          <div class="journal-kicker">Research Journal</div>
          <h1 class="journal-title">{}</h1>
          <div class="journal-meta">
            <span class="domain-pill">{}</span>
            <span class="run-since">Since {run_ts}</span>
          </div>
        </header>"#,
        esc(&program_title),
        esc(&program_domain)
    ));

    // ── Stats strip ──
    body.push_str(&format!(
        r#"<div class="stats-strip">
          <div class="stat-cell"><div class="stat-n">{}</div><div class="stat-l">questions</div></div>
          <div class="stat-cell"><div class="stat-n">{}</div><div class="stat-l">hypotheses</div></div>
          <div class="stat-cell"><div class="stat-n">{}</div><div class="stat-l">tested</div></div>
          <div class="stat-cell stat-ok"><div class="stat-n">{n_succeeded}</div><div class="stat-l">confirmed</div></div>
          <div class="stat-cell stat-fail"><div class="stat-n">{n_failed}</div><div class="stat-l">refuted</div></div>
          <div class="stat-cell"><div class="stat-n">{}</div><div class="stat-l">claims</div></div>
        </div>"#,
        questions.len(),
        hypotheses.len(),
        n_with_post,
        all_claims.len()
    ));

    // ── Research Threads ──
    body.push_str(r#"<section class="journal-section" id="threads">
      <h2 class="section-heading">Research Threads</h2>"#);

    if hypotheses.is_empty() {
        body.push_str(r#"<div class="empty-state">No hypotheses recorded yet.</div>"#);
    }

    for q_text in &ordered_questions {
        let hyps = match question_to_hyps.get(q_text) {
            Some(h) => h,
            None => continue,
        };

        body.push_str(&format!(
            r#"<details class="thread-block" open>
              <summary class="thread-summary">
                <span class="thread-q-mark">?</span>
                <span class="thread-q-text">{}</span>
                <span class="thread-count">{} hypothesis{}</span>
              </summary>
              <div class="thread-entries">"#,
            esc(q_text),
            hyps.len(),
            if hyps.len() == 1 { "" } else { "es" }
        ));

        for hyp in hyps {
            let q_text_for_entry = hyp_to_question.get(hyp.id).copied().unwrap_or("");
            body.push_str(&render_hypothesis_entry(
                hyp, q_text_for_entry,
                out_edges, in_edges, node_map, artifacts_dir
            ));
        }

        body.push_str("</div></details>");
    }

    // Ungrouped hypotheses
    if !ungrouped_hyps.is_empty() {
        body.push_str(&format!(
            r#"<details class="thread-block" open>
              <summary class="thread-summary">
                <span class="thread-q-mark">∅</span>
                <span class="thread-q-text">Uncategorized hypotheses</span>
                <span class="thread-count">{}</span>
              </summary>
              <div class="thread-entries">"#,
            ungrouped_hyps.len()
        ));
        for hyp in &ungrouped_hyps {
            body.push_str(&render_hypothesis_entry(
                hyp, "",
                out_edges, in_edges, node_map, artifacts_dir
            ));
        }
        body.push_str("</div></details>");
    }

    body.push_str("</section>");

    // ── Findings grid ──
    if !confirmed_claims.is_empty() {
        body.push_str(r#"<section class="journal-section" id="findings">
          <h2 class="section-heading">Findings</h2>
          <div class="findings-grid">"#);

        for claim in &confirmed_claims {
            let cd = claim.data;
            let stmt = s(cd, "statement");
            let conf = cd["confidence"].as_f64().unwrap_or(0.0);
            let color = confidence_color(conf);
            body.push_str(&format!(
                r#"<div class="finding-card" style="border-top-color:{color}">
                  <div class="finding-icon">&#x2713;</div>
                  <div class="finding-text">{}</div>
                  {}
                </div>"#,
                esc(&stmt),
                render_confidence_bar(conf, 100)
            ));
        }

        body.push_str("</div></section>");
    }

    // ── Reverted runs ──
    if !failed_runs.is_empty() {
        body.push_str(&format!(
            r#"<section class="journal-section" id="reverted">
              <h2 class="section-heading">Tried &amp; Rejected</h2>
              <p class="section-intro">These experiments ran but did not pass the threshold. Recorded for transparency.</p>
              <details class="reverted-block">
                <summary class="reverted-summary">{} failed run{} — expand to review</summary>
                <div class="reverted-list">"#,
            failed_runs.len(),
            if failed_runs.len() == 1 { "" } else { "s" }
        ));

        for run in &failed_runs {
            let rd = run.data;
            let started = fmt_timestamp(rd["started_at"].as_str().unwrap_or(""));
            let dir_name = run_dir_name(rd["started_at"].as_str().unwrap_or(""));
            let artifact = if !dir_name.is_empty() { load_artifact(artifacts_dir, &dir_name) } else { None };

            // Find the plan and hypothesis for this run
            let plan_nodes = out_edges.get(run.id)
                .map(|edges| {
                    edges.iter()
                        .filter(|(_, ek)| ek == "DerivedFrom")
                        .filter_map(|(pid, _)| node_map.get(pid.as_str()))
                        .filter(|n| n.kind == "ExperimentPlan")
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();

            let hyp_stmt = plan_nodes.first()
                .and_then(|plan| out_edges.get(plan.id))
                .and_then(|edges| {
                    edges.iter()
                        .find(|(_, ek)| ek == "DerivedFrom")
                        .and_then(|(hid, _)| node_map.get(hid.as_str()))
                        .filter(|n| n.kind == "Hypothesis")
                        .map(|h| s(h.data, "statement"))
                })
                .unwrap_or_default();

            body.push_str(&format!(
                r#"<div class="reverted-entry">
                  <div class="reverted-date">{started}</div>"#
            ));

            if !hyp_stmt.is_empty() {
                body.push_str(&format!(
                    r#"<div class="reverted-hyp"><s>{}</s></div>"#,
                    esc(&hyp_stmt)
                ));
            }

            if let Some(art) = &artifact {
                if !art.summary.is_empty() {
                    body.push_str(&format!(
                        r#"<div class="reverted-reason">{}</div>"#,
                        esc(&art.summary)
                    ));
                }
            }

            body.push_str("</div>");
        }

        body.push_str("</div></details></section>");
    }

    // ── Graph JSON ──
    let gj = serde_json::to_string_pretty(graph).unwrap_or_default();
    body.push_str(&format!(
        r#"<section class="journal-section" id="graph">
          <h2 class="section-heading">Belief Graph</h2>
          <details class="json-block">
            <summary class="json-summary">Raw graph_state.json ({} nodes, {} edges)</summary>
            <pre class="graph-json">{}</pre>
          </details>
        </section>"#,
        graph["nodes"].as_array().map(|a| a.len()).unwrap_or(0),
        graph["edges"].as_array().map(|a| a.len()).unwrap_or(0),
        esc(&gj)
    ));

    wrap_page(&body, &program_title)
}

// ── Page wrapper ──────────────────────────────────────────────────────────────

fn wrap_page(body: &str, title: &str) -> String {
    format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — Research Journal</title>
<style>
/* ── Reset ── */
*{{box-sizing:border-box;margin:0;padding:0}}
html{{font-size:15px;scroll-behavior:smooth}}
body{{
  background:#f4efe6;
  color:#1c1713;
  font-family:'Georgia','Times New Roman',serif;
  line-height:1.7;
  min-height:100vh;
}}
strong{{font-weight:700}}
em{{font-style:italic}}
code,pre{{font-family:'JetBrains Mono','Fira Mono',Menlo,monospace}}

/* ── Nav ── */
nav{{
  position:sticky;top:0;z-index:200;
  background:rgba(244,239,230,.95);
  backdrop-filter:blur(12px);
  border-bottom:1px solid #d8cfc0;
  padding:.6rem 2.5rem;
  display:flex;align-items:center;gap:2rem;
}}
.nav-brand{{font-size:.8rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
  color:#1c1713;text-decoration:none;opacity:.5}}
.nav-brand:hover{{opacity:1}}
.nav-links{{margin-left:auto;display:flex;gap:1.8rem}}
.nav-links a{{font-size:.75rem;color:#1c1713;opacity:.45;text-decoration:none;
  letter-spacing:.04em;transition:opacity .15s}}
.nav-links a:hover{{opacity:1}}

/* ── Layout ── */
.page{{max-width:860px;margin:0 auto;padding:3rem 2rem 6rem}}

/* ── Journal header ── */
.journal-header{{margin-bottom:2.5rem;padding-bottom:2.5rem;border-bottom:2px solid #1c1713}}
.journal-kicker{{font-size:.65rem;font-weight:700;letter-spacing:.18em;text-transform:uppercase;
  color:#8a5c00;margin-bottom:.7rem}}
.journal-title{{font-size:2.4rem;font-weight:700;letter-spacing:-.04em;line-height:1.15;
  margin-bottom:.9rem;color:#1c1713}}
.journal-meta{{display:flex;align-items:center;gap:1rem;flex-wrap:wrap}}
.domain-pill{{font-size:.68rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
  background:#1c1713;color:#f4efe6;padding:.25rem .65rem;border-radius:2px}}
.run-since{{font-size:.78rem;color:#9a8f7c;font-style:italic}}

/* ── Stats strip ── */
.stats-strip{{
  display:grid;grid-template-columns:repeat(6,1fr);
  border:1px solid #d8cfc0;border-radius:4px;overflow:hidden;
  margin-bottom:3.5rem;background:#fff;
}}
.stat-cell{{
  padding:1.1rem .8rem;text-align:center;
  border-right:1px solid #d8cfc0;
}}
.stat-cell:last-child{{border-right:none}}
.stat-ok{{background:rgba(45,106,79,.06)}}
.stat-fail{{background:rgba(185,64,64,.06)}}
.stat-n{{font-size:1.8rem;font-weight:700;letter-spacing:-.05em;line-height:1;font-family:Georgia,serif}}
.stat-ok .stat-n{{color:#2d6a4f}}
.stat-fail .stat-n{{color:#b94040}}
.stat-l{{font-size:.6rem;text-transform:uppercase;letter-spacing:.1em;color:#9a8f7c;margin-top:.3rem}}
@media(max-width:640px){{
  .stats-strip{{grid-template-columns:repeat(3,1fr)}}
  .stat-cell:nth-child(3){{border-right:none}}
}}

/* ── Section headings ── */
.journal-section{{margin-bottom:4rem}}
.section-heading{{
  font-size:1rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
  color:#8a5c00;margin-bottom:1.75rem;padding-bottom:.65rem;
  border-bottom:1px solid #d8cfc0;
}}
.section-intro{{font-size:.85rem;color:#7a6f5e;font-style:italic;margin-bottom:1.25rem;margin-top:-.5rem}}

/* ── Thread blocks ── */
.thread-block{{margin-bottom:1.5rem;border:1px solid #d8cfc0;border-radius:5px;overflow:hidden}}
.thread-summary{{
  display:flex;align-items:baseline;gap:.75rem;flex-wrap:wrap;
  padding:1rem 1.4rem;background:#fff;cursor:pointer;
  list-style:none;user-select:none;
}}
.thread-summary::-webkit-details-marker{{display:none}}
details[open] .thread-summary{{border-bottom:1px solid #d8cfc0}}
.thread-q-mark{{
  font-size:1.1rem;font-weight:700;color:#8a5c00;
  background:rgba(138,92,0,.1);width:1.6rem;height:1.6rem;
  display:inline-flex;align-items:center;justify-content:center;
  border-radius:50%;flex-shrink:0;margin-top:.05rem;
}}
.thread-q-text{{font-size:.97rem;font-weight:600;letter-spacing:-.015em;flex:1}}
.thread-count{{font-size:.7rem;color:#9a8f7c;margin-left:auto;white-space:nowrap}}
.thread-entries{{padding:1.25rem 1.4rem;background:#faf7f2;display:flex;flex-direction:column;gap:1.5rem}}

/* ── Journal entry ── */
.journal-entry{{
  display:grid;grid-template-columns:80px 1fr;gap:1.25rem;
  background:#fff;border:1px solid #d8cfc0;border-radius:4px;overflow:hidden;
}}
.journal-pending{{opacity:.75}}
.entry-gutter{{
  background:#faf7f2;border-right:1px solid #e8e0d0;
  padding:1.1rem .8rem;display:flex;flex-direction:column;align-items:center;gap:.5rem;
}}
.entry-tier{{
  font-size:.65rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
  color:#8a5c00;background:rgba(138,92,0,.12);padding:.2rem .4rem;border-radius:2px;
}}
.entry-date{{font-size:.6rem;color:#9a8f7c;text-align:center;line-height:1.4}}
.entry-body{{padding:1.25rem 1.4rem 1.25rem 0}}

/* ── Entry question ── */
.entry-question{{
  font-size:.78rem;color:#7a6f5e;font-style:italic;
  margin-bottom:.9rem;padding:.5rem .75rem;
  background:rgba(138,92,0,.06);border-left:2px solid #c4a96a;
  border-radius:0 3px 3px 0;
}}
.question-mark{{font-weight:700;color:#8a5c00;margin-right:.4rem}}

/* ── Hypothesis ── */
.entry-hypothesis{{margin-bottom:1.25rem}}
.hyp-statement{{
  font-size:1.05rem;font-weight:600;letter-spacing:-.02em;line-height:1.45;
  margin-bottom:1rem;color:#1c1713;
}}
.hyp-priors{{display:flex;flex-wrap:wrap;gap:1.25rem}}
.prior-block,.posterior-block{{display:flex;align-items:center;gap:.75rem;flex-wrap:wrap}}
.prior-label{{font-size:.65rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
  color:#9a8f7c;min-width:52px}}
.delta-up{{font-size:.72rem;color:#2d6a4f;font-weight:600}}
.delta-down{{font-size:.72rem;color:#b94040;font-weight:600}}

/* ── Confidence meter ── */
.conf-meter{{display:flex;align-items:center;gap:.6rem}}
.conf-track{{background:#e8e0d0;border-radius:2px;height:6px;flex-shrink:0}}
.conf-fill{{height:100%;border-radius:2px;transition:width .3s}}
.conf-value{{font-size:.75rem;font-weight:600;font-family:monospace;white-space:nowrap}}
.conf-label{{font-size:.65rem;font-weight:400;opacity:.7;font-style:normal}}

/* ── Experiment block ── */
.experiment-block{{
  border-top:1px solid #e8e0d0;padding-top:1.1rem;margin-bottom:1.1rem;
}}
.exp-label{{font-size:.6rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
  color:#9a8f7c;margin-bottom:.4rem}}
.exp-description{{font-size:.85rem;color:#5a5044;font-style:italic;margin-bottom:.8rem}}
.steps-label{{font-size:.6rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
  color:#9a8f7c;margin-bottom:.4rem;margin-top:.75rem}}

/* ── Plan steps ── */
.plan-steps{{
  padding-left:1.25rem;margin-bottom:.75rem;
}}
.step-item{{font-size:.82rem;color:#5a5044;margin-bottom:.3rem;line-height:1.5}}

/* ── Run block ── */
.run-block{{
  border-radius:4px;padding:1rem 1.1rem;margin-top:.75rem;
  border:1px solid #d8cfc0;
}}
.verdict-confirmed{{background:rgba(45,106,79,.05);border-color:rgba(45,106,79,.3)}}
.verdict-refuted{{background:rgba(185,64,64,.04);border-color:rgba(185,64,64,.25)}}
.run-header{{display:flex;align-items:center;gap:1rem;margin-bottom:.9rem;flex-wrap:wrap}}
.run-verdict-badge{{
  font-size:.65rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
  padding:.25rem .65rem;border-radius:2px;
}}
.verdict-confirmed .run-verdict-badge{{background:rgba(45,106,79,.15);color:#2d6a4f}}
.verdict-refuted .run-verdict-badge{{background:rgba(185,64,64,.1);color:#b94040}}
.run-meta{{font-size:.7rem;color:#9a8f7c;margin-left:auto}}

/* ── Debate ── */
.debate-block{{margin:1rem 0}}
.debate-heading{{font-size:.6rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
  color:#9a8f7c;margin-bottom:.6rem}}
.debate-turn{{
  border-left:3px solid #d8cfc0;padding:.65rem .85rem;margin-bottom:.5rem;border-radius:0 3px 3px 0;
}}
.debate-critic{{border-left-color:#b94040;background:rgba(185,64,64,.04)}}
.debate-empiricist{{border-left-color:#2a4d8f;background:rgba(42,77,143,.04)}}
.debate-synthesizer{{border-left-color:#2d6a4f;background:rgba(45,106,79,.04)}}
.debate-other{{border-left-color:#8a5c00;background:rgba(138,92,0,.04)}}
.debate-speaker{{
  font-size:.6rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
  margin-bottom:.3rem;
}}
.debate-critic .debate-speaker{{color:#b94040}}
.debate-empiricist .debate-speaker{{color:#2a4d8f}}
.debate-synthesizer .debate-speaker{{color:#2d6a4f}}
.debate-other .debate-speaker{{color:#8a5c00}}
.debate-text{{font-size:.82rem;color:#3a3028;line-height:1.6;white-space:pre-wrap}}
.debate-absent{{font-size:.78rem;color:#9a8f7c;font-style:italic;margin:.75rem 0}}

/* ── Observations ── */
.observations{{margin:1rem 0}}
.obs-heading{{font-size:.6rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
  color:#9a8f7c;margin-bottom:.5rem}}
.obs-item{{
  font-size:.82rem;color:#3a3028;padding:.4rem .7rem;
  border-left:2px solid #d8cfc0;margin-bottom:.3rem;line-height:1.5;
  background:rgba(255,255,255,.6);border-radius:0 2px 2px 0;
}}
.obs-long{{cursor:pointer}}
.obs-long summary{{font-size:.82rem;color:#3a3028;padding:.35rem .7rem;list-style:none;
  border-left:2px solid #c0b89e;background:rgba(255,255,255,.5)}}
.obs-long summary::-webkit-details-marker{{display:none}}
.obs-long[open] summary{{border-left-color:#8a5c00}}
.obs-full{{font-size:.8rem;color:#5a5044;padding:.65rem .85rem;
  border-left:2px solid #c0b89e;background:#faf7f2;white-space:pre-wrap;line-height:1.55}}

/* ── Summary ── */
.run-summary{{margin:1rem 0}}
.summary-label{{font-size:.6rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
  color:#9a8f7c;margin-bottom:.4rem}}
.summary-text{{font-size:.85rem;color:#3a3028;line-height:1.65;font-style:italic;
  padding:.7rem 1rem;background:rgba(138,92,0,.05);border-left:3px solid #c4a96a;
  border-radius:0 3px 3px 0}}

/* ── Metrics ── */
.metrics-block{{margin:.75rem 0}}
.metrics-block summary{{font-size:.75rem;color:#7a6f5e;cursor:pointer;padding:.3rem 0;list-style:none}}
.metrics-block summary::-webkit-details-marker{{display:none}}
.metrics-table{{margin-top:.5rem;border-collapse:collapse;width:100%;font-size:.78rem}}
.metric-key{{padding:.25rem .5rem .25rem 0;color:#7a6f5e;border-bottom:1px solid #e8e0d0;
  font-family:monospace;white-space:nowrap;width:40%}}
.metric-val{{padding:.25rem 0 .25rem .5rem;color:#1c1713;border-bottom:1px solid #e8e0d0;
  font-family:monospace}}

/* ── Fitness chart ── */
.fitness-label{{font-size:.6rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
  color:#9a8f7c;margin:.9rem 0 .5rem}}
.fitness-chart{{display:flex;gap:.5rem;align-items:flex-end;height:70px;
  padding:.4rem .4rem 0;background:rgba(255,255,255,.5);border-radius:3px;border:1px solid #e8e0d0}}
.fitness-cell{{display:flex;flex-direction:column;align-items:center;flex:1;gap:.2rem}}
.fitness-bar-wrap{{flex:1;width:100%;display:flex;align-items:flex-end;min-height:40px}}
.fitness-bar{{width:100%;border-radius:1px 1px 0 0;min-height:2px}}
.fitness-name{{font-size:.48rem;text-align:center;color:#9a8f7c;line-height:1.2;letter-spacing:0}}
.fitness-val{{font-size:.6rem;font-weight:700;font-family:monospace}}

/* ── Entry claims ── */
.entry-claims{{margin-top:1rem;padding-top:1rem;border-top:1px solid #e8e0d0}}
.claim-finding{{
  padding:.75rem .9rem;margin-bottom:.5rem;
  background:rgba(45,106,79,.04);border:1px solid rgba(45,106,79,.2);
  border-radius:3px;
}}
.claim-finding-text{{font-size:.83rem;color:#1c1713;margin-bottom:.5rem;line-height:1.5}}
.claim-conf-row{{display:flex;align-items:center}}

/* ── Findings grid ── */
.findings-grid{{
  display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:1rem;
}}
.finding-card{{
  background:#fff;border:1px solid #d8cfc0;border-radius:4px;
  border-top:3px solid #2d6a4f;
  padding:1.1rem 1.25rem;
}}
.finding-icon{{font-size:1.1rem;color:#2d6a4f;margin-bottom:.5rem}}
.finding-text{{font-size:.88rem;color:#1c1713;line-height:1.55;margin-bottom:.75rem;font-weight:500}}

/* ── Reverted ── */
.reverted-block{{border:1px solid #d8cfc0;border-radius:4px;overflow:hidden}}
.reverted-summary{{
  padding:.85rem 1.25rem;background:#fff;cursor:pointer;list-style:none;
  font-size:.82rem;color:#7a6f5e;font-style:italic;
}}
.reverted-summary::-webkit-details-marker{{display:none}}
.reverted-list{{padding:1rem 1.25rem;background:#faf7f2;display:flex;flex-direction:column;gap:.75rem}}
.reverted-entry{{
  padding:.75rem 1rem;border:1px solid #e8e0d0;border-radius:3px;
  background:rgba(185,64,64,.03);opacity:.7;
}}
.reverted-date{{font-size:.65rem;color:#9a8f7c;margin-bottom:.4rem;font-family:monospace}}
.reverted-hyp{{font-size:.85rem;color:#9a8f7c;margin-bottom:.35rem;text-decoration:line-through;
  text-decoration-color:rgba(185,64,64,.5)}}
.reverted-reason{{font-size:.78rem;color:#7a6f5e;font-style:italic;line-height:1.55}}

/* ── Graph JSON ── */
.json-block{{border:1px solid #d8cfc0;border-radius:4px;overflow:hidden}}
.json-summary{{
  padding:.85rem 1.25rem;background:#fff;cursor:pointer;list-style:none;
  font-size:.82rem;color:#7a6f5e;
}}
.json-summary::-webkit-details-marker{{display:none}}
.graph-json{{
  font-size:.68rem;line-height:1.5;
  background:#1c1713;color:#c8bda8;
  padding:1.5rem;overflow:auto;max-height:500px;white-space:pre;
}}

/* ── Empty state ── */
.empty-state{{font-size:.88rem;color:#9a8f7c;font-style:italic;padding:1.5rem 0}}
</style>
</head>
<body>
<nav>
  <a href="#" class="nav-brand">chitta-research</a>
  <div class="nav-links">
    <a href="#threads">Threads</a>
    <a href="#findings">Findings</a>
    <a href="#reverted">Rejected</a>
    <a href="#graph">Graph</a>
  </div>
</nav>
<div class="page">
{body}
</div>
</body>
</html>"##,
        title = esc(title),
        body = body
    )
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();

    let json_str = fs::read_to_string(&cli.graph)
        .with_context(|| format!("Cannot read {}", cli.graph.display()))?;
    let graph: Value = serde_json::from_str(&json_str)
        .with_context(|| format!("Invalid JSON in {}", cli.graph.display()))?;

    let (out_edges, in_edges) = build_edge_maps(&graph);

    // Build typed node map
    let raw_nodes = graph["nodes"].as_array().cloned().unwrap_or_default();
    let typed_nodes: Vec<(String, String, &Value)> = raw_nodes.iter()
        .filter_map(|n| {
            let id = n["id"].as_str()?.to_string();
            let kind_obj = n["kind"].as_object()?;
            let kind = kind_obj.keys().next()?.clone();
            Some((id, kind, n))
        })
        .collect();

    // We need a lifetime-safe map: borrow from the original graph value
    // Re-build typed nodes directly from the graph's array reference
    let nodes_arr = graph["nodes"].as_array().map(|a| a.as_slice()).unwrap_or(&[]);
    let mut node_map: HashMap<&str, TypedNode<'_>> = HashMap::new();
    for n in nodes_arr {
        if let Some(tn) = typed(n) {
            node_map.insert(tn.id, tn);
        }
    }
    drop(typed_nodes); // not needed anymore

    let html = render(&graph, &out_edges, &in_edges, &node_map, &cli.artifacts);

    fs::write(&cli.output, &html)
        .with_context(|| format!("Cannot write {}", cli.output.display()))?;

    let n_hyps = node_map.values().filter(|n| n.kind == "Hypothesis").count();
    let n_runs = node_map.values().filter(|n| n.kind == "Run").count();
    let n_claims = node_map.values().filter(|n| n.kind == "Claim").count();
    println!("Report written to {}", cli.output.display());
    println!("  {} hypotheses  {} runs  {} claims", n_hyps, n_runs, n_claims);

    Ok(())
}
