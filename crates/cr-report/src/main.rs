/// cr-report — generate a self-contained HTML exploration report from graph_state.json
///
/// Usage:
///   cr-report --graph graph_state.json --artifacts ./artifacts --output report.html
///   cr-report   # uses defaults in current directory

use clap::Parser;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "cr-report", about = "Generate HTML report from chitta-research graph")]
struct Cli {
    #[arg(long, default_value = "graph_state.json")]
    graph: PathBuf,
    #[arg(long, default_value = "artifacts")]
    artifacts: PathBuf,
    #[arg(long, default_value = "report.html")]
    output: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let json_str = fs::read_to_string(&cli.graph)
        .map_err(|e| anyhow::anyhow!("Cannot read {}: {}", cli.graph.display(), e))?;
    let graph: Value = serde_json::from_str(&json_str)?;
    let nodes = graph["nodes"].as_array().cloned().unwrap_or_default();

    let mut programs: Vec<&Value> = vec![];
    let mut questions:  Vec<&Value> = vec![];
    let mut hypotheses: Vec<&Value> = vec![];
    let mut plans:      Vec<&Value> = vec![];
    let mut runs:       Vec<&Value> = vec![];
    let mut observations: Vec<&Value> = vec![];
    let mut claims:     Vec<&Value> = vec![];

    for n in &nodes {
        match node_kind(n) {
            "ResearchProgram" => programs.push(n),
            "Question"        => questions.push(n),
            "Hypothesis"      => hypotheses.push(n),
            "ExperimentPlan"  => plans.push(n),
            "Run"             => runs.push(n),
            "Observation"     => observations.push(n),
            "Claim"           => claims.push(n),
            _ => {}
        }
    }

    // Build node lookup by id
    let node_map: HashMap<String, &Value> = nodes.iter()
        .map(|n| (get_str(n, "id"), n))
        .collect();

    // Edges: source -> [(target, kind)]
    let edges = graph["edges"].as_array().cloned().unwrap_or_default();
    let mut out_edges: HashMap<String, Vec<(String, String)>> = HashMap::new();
    for e in &edges {
        let src = e["source"].as_str().unwrap_or("").to_string();
        let dst = e["target"].as_str().unwrap_or("").to_string();
        let kind = e["kind"].as_str().unwrap_or("").to_string();
        out_edges.entry(src).or_default().push((dst, kind));
    }

    // Artifact files keyed by run dir name
    let mut run_artifacts: HashMap<String, Value> = HashMap::new();
    if cli.artifacts.exists() {
        for entry in fs::read_dir(&cli.artifacts).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                let results = path.join("results.json");
                if let Ok(s) = fs::read_to_string(&results) {
                    if let Ok(v) = serde_json::from_str::<Value>(&s) {
                        let name = path.file_name().unwrap_or_default()
                            .to_string_lossy().to_string();
                        run_artifacts.insert(name, v);
                    }
                }
            }
        }
    }

    let total_nodes   = nodes.len();
    let total_edges   = edges.len();
    let total_actions = graph["total_actions"].as_u64().unwrap_or(0);

    let html = render(&programs, &questions, &hypotheses, &plans, &runs,
                      &observations, &claims, &run_artifacts, &node_map,
                      &out_edges, total_nodes, total_edges, total_actions, &graph);
    fs::write(&cli.output, &html)?;
    println!("Report written to {}", cli.output.display());
    println!("  {} programs  {} hypotheses  {} runs  {} claims",
        programs.len(), hypotheses.len(), runs.len(), claims.len());
    Ok(())
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn node_kind(n: &Value) -> &str {
    n["kind"].as_object()
        .and_then(|o| o.keys().next())
        .map(|s| s.as_str())
        .unwrap_or("")
}

fn kind_data(n: &Value) -> &Value {
    n["kind"].as_object()
        .and_then(|o| o.values().next())
        .unwrap_or(&Value::Null)
}

fn get_str(v: &Value, key: &str) -> String {
    v[key].as_str().unwrap_or("").to_string()
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

fn confidence_class(c: f64) -> &'static str {
    if c >= 0.75 { "conf-high" } else if c >= 0.5 { "conf-mid" } else { "conf-low" }
}

fn status_class(s: &str) -> &'static str {
    if s == "Succeeded" { "ok" } else if s == "Failed" { "fail" } else { "pending" }
}

// ── Renderer ─────────────────────────────────────────────────────────────────

fn render(
    programs: &[&Value],
    questions: &[&Value],
    hypotheses: &[&Value],
    plans: &[&Value],
    runs: &[&Value],
    _observations: &[&Value],
    claims: &[&Value],
    artifacts: &HashMap<String, Value>,
    node_map: &HashMap<String, &Value>,
    out_edges: &HashMap<String, Vec<(String, String)>>,
    total_nodes: usize,
    _total_edges: usize,
    total_actions: u64,
    graph: &Value,
) -> String {
    let mut b = String::with_capacity(64 * 1024);

    // Stats
    let succeeded = runs.iter().filter(|r| get_str(kind_data(r), "status") == "Succeeded").count();
    let failed    = runs.iter().filter(|r| get_str(kind_data(r), "status") == "Failed").count();

    b.push_str(&format!(r#"
<div class="stats-row">
  <div class="stat"><div class="stat-n">{}</div><div class="stat-l">programs</div></div>
  <div class="stat"><div class="stat-n">{}</div><div class="stat-l">questions</div></div>
  <div class="stat"><div class="stat-n">{}</div><div class="stat-l">hypotheses</div></div>
  <div class="stat"><div class="stat-n"><span class="c-ok">{}</span> / <span class="c-fail">{}</span></div><div class="stat-l">runs (ok/fail)</div></div>
  <div class="stat"><div class="stat-n">{}</div><div class="stat-l">claims</div></div>
  <div class="stat"><div class="stat-n">{}</div><div class="stat-l">graph nodes</div></div>
</div>"#,
        programs.len(), questions.len(), hypotheses.len(),
        succeeded, failed, claims.len(), total_nodes));

    // ── Programs ──
    b.push_str(r#"<section><div class="sec-label">Research Programs</div>"#);
    for p in programs {
        let d = kind_data(p);
        let title  = get_str(d, "title");
        let domain = get_str(d, "domain");
        let prio   = d["priority"].as_f64().unwrap_or(0.0);
        // Questions for this program
        let pid = get_str(p, "id");
        let q_ids: Vec<String> = out_edges.get(&pid)
            .map(|es| es.iter().filter(|(_, k)| k == "DerivedFrom").map(|(id, _)| id.clone()).collect())
            .unwrap_or_default();

        b.push_str(&format!(r#"<div class="program-block">
<div class="program-title"><span class="domain-tag">{domain}</span>{title}<span class="prio">priority {prio:.1}</span></div>"#));

        // Show questions under program
        b.push_str(r#"<div class="question-list">"#);
        for n in questions {
            let nd = kind_data(n);
            let text = get_str(nd, "text");
            if !text.is_empty() {
                b.push_str(&format!(r#"<div class="q-item">Q: {}</div>"#, esc(&text)));
            }
        }
        b.push_str("</div></div>");
    }
    b.push_str("</section>");

    // ── Hypothesis timeline ──
    b.push_str(r#"<section><div class="sec-label">Hypothesis → Experiment → Finding</div><div class="timeline">"#);

    for h in hypotheses {
        let d     = kind_data(h);
        let stmt  = get_str(d, "statement");
        let prior = d["prior_confidence"].as_f64().unwrap_or(0.0);
        let post  = d["posterior_confidence"].as_f64();
        let hid   = get_str(h, "id");

        let conf_display = match post {
            Some(p) => format!(r#"<span class="{}">prior {:.2} → posterior {:.2}</span>"#, confidence_class(p), prior, p),
            None    => format!(r#"<span class="conf-mid">prior {:.2} (untested)</span>"#, prior),
        };

        // Find plans for this hypothesis
        let plan_ids: Vec<String> = out_edges.get(&hid)
            .map(|es| es.iter().filter(|(_, k)| k == "DerivedFrom" || k == "Supports")
                .map(|(id, _)| id.clone()).collect())
            .unwrap_or_default();

        let mut plans_html = String::new();
        for plan_id in &plan_ids {
            let Some(pn) = node_map.get(plan_id) else { continue };
            if node_kind(pn) != "ExperimentPlan" { continue }
            let pd = kind_data(pn);
            let desc = get_str(pd, "description");
            let steps = pd["steps"].as_array()
                .map(|a| a.iter().map(|s| format!(r#"<li>{}</li>"#, esc(s.as_str().unwrap_or("")))).collect::<Vec<_>>().join(""))
                .unwrap_or_default();

            // Find run for this plan
            let run_ids: Vec<String> = out_edges.get(plan_id)
                .map(|es| es.iter().filter(|(_, k)| k == "DerivedFrom")
                    .map(|(id, _)| id.clone()).collect())
                .unwrap_or_default();

            let mut run_html = String::new();
            for run_id in &run_ids {
                let Some(rn) = node_map.get(run_id) else { continue };
                if node_kind(rn) != "Run" { continue }
                let rd = kind_data(rn);
                let status = get_str(rd, "status");
                let commit = get_str(rd, "artifact_commit");
                let sc = status_class(&status);

                // Find artifact content
                let art_html = artifacts.iter()
                    .find(|(name, _)| {
                        let c8 = &commit[..commit.len().min(12)];
                        name.contains(c8) || commit.is_empty()
                    })
                    .map(|(_, v)| {
                        let obs = v["observations"].as_array()
                            .map(|a| a.iter().enumerate().map(|(i, o)| {
                                let s = o.as_str().unwrap_or("");
                                // Multi-line observations get a <details>
                                if s.contains('\n') || s.len() > 120 {
                                    format!(r#"<details class="obs-detail"><summary>observation {}</summary><pre>{}</pre></details>"#, i+1, esc(&s[..s.len().min(3000)]))
                                } else {
                                    format!(r#"<div class="obs-line">{}</div>"#, esc(s))
                                }
                            }).collect::<Vec<_>>().join(""))
                            .unwrap_or_default();
                        let summary = get_str(v, "summary");
                        format!(r#"<div class="run-artifact"><div class="art-summary">{}</div>{}</div>"#,
                            esc(&summary), obs)
                    })
                    .unwrap_or_default();

                run_html.push_str(&format!(r#"<div class="run-row"><span class="run-status {sc}">{status}</span><code class="run-id-sm">{}</code>{}</div>"#,
                    &commit[..commit.len().min(12)], art_html));
            }

            plans_html.push_str(&format!(r#"
<div class="plan-block">
  <div class="plan-desc">{}</div>
  <ol class="plan-steps">{steps}</ol>
  {run_html}
</div>"#, esc(&desc)));
        }

        // Find claims derived from this hypothesis
        let claim_ids: Vec<String> = out_edges.get(&hid)
            .map(|es| es.iter().filter(|(_, k)| k == "Supports" || k == "DerivedFrom")
                .map(|(id, _)| id.clone()).collect())
            .unwrap_or_default();

        let mut claims_html = String::new();
        for cid in &claim_ids {
            let Some(cn) = node_map.get(cid) else { continue };
            if node_kind(cn) != "Claim" { continue }
            let cd = kind_data(cn);
            let content = get_str(cd, "content");
            let conf = cd["confidence"].as_f64().unwrap_or(0.0);
            claims_html.push_str(&format!(
                r#"<div class="claim-chip {}">&#x2713; {} <span class="chip-conf">{:.2}</span></div>"#,
                confidence_class(conf), esc(&content), conf));
        }

        b.push_str(&format!(r#"
<div class="timeline-item">
  <div class="tl-dot"></div>
  <div class="tl-body">
    <div class="hyp-stmt">{}</div>
    <div class="hyp-conf">{conf_display}</div>
    {plans_html}
    {claims_html}
  </div>
</div>"#, esc(&stmt)));
    }

    if hypotheses.is_empty() {
        b.push_str(r#"<div class="empty-state">No hypotheses yet — Hotr is still running or the run was too short.</div>"#);
    }

    b.push_str("</div></section>");

    // ── Claims summary ──
    if !claims.is_empty() {
        b.push_str(r#"<section><div class="sec-label">All Claims</div><div class="claims-grid">"#);
        for c in claims {
            let d = kind_data(c);
            let content = get_str(d, "content");
            let conf    = d["confidence"].as_f64().unwrap_or(0.0);
            let cc      = confidence_class(conf);
            b.push_str(&format!(r#"<div class="claim-card {cc}">
  <div class="claim-text">{}</div>
  <div class="claim-conf"><div class="conf-bar" style="width:{:.0}%"></div><span>{:.2}</span></div>
</div>"#, esc(&content), conf * 100.0, conf));
        }
        b.push_str("</div></section>");
    }

    // ── Graph JSON ──
    let gj = serde_json::to_string_pretty(graph).unwrap_or_default();
    b.push_str(&format!(r#"<section class="json-section">
  <div class="sec-label">Belief Graph JSON <button class="toggle-btn" onclick="toggleJson()">show / hide</button></div>
  <pre id="graph-json" class="graph-json hidden">{}</pre>
</section>"#, esc(&gj)));

    wrap_page(&b)
}

fn wrap_page(body: &str) -> String {
    format!(r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>chitta-research report</title>
<style>
/* ── Reset & base ── */
*{{box-sizing:border-box;margin:0;padding:0}}
html{{font-size:15px}}
body{{
  background:#f4efe6;
  color:#1a1712;
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
  line-height:1.65;
  min-height:100vh;
}}
a{{color:#2a4d8f;text-underline-offset:3px}}
code,pre{{font-family:'JetBrains Mono','Fira Code',Menlo,monospace}}

/* ── Nav ── */
nav{{
  background:rgba(244,239,230,.92);
  backdrop-filter:blur(10px);
  border-bottom:1px solid #d0c5b0;
  padding:.7rem 2rem;
  display:flex;align-items:center;gap:2rem;
  position:sticky;top:0;z-index:100;
}}
.nav-logo{{font-weight:700;font-size:.92rem;letter-spacing:-.02em;color:#1a1712;text-decoration:none}}
.nav-links{{display:flex;gap:1.5rem;margin-left:auto}}
.nav-links a{{font-size:.78rem;color:#1a1712;opacity:.55;text-decoration:none;transition:opacity .15s}}
.nav-links a:hover{{opacity:1}}

/* ── Layout ── */
.page{{max-width:900px;margin:0 auto;padding:2.5rem 2rem 5rem}}
section{{margin-bottom:3.5rem}}

/* ── Stats row ── */
.stats-row{{
  display:grid;
  grid-template-columns:repeat(6,1fr);
  gap:0;
  border:1px solid #d0c5b0;
  border-radius:6px;
  overflow:hidden;
  margin-bottom:3rem;
}}
.stat{{
  background:#fff;
  border-right:1px solid #d0c5b0;
  padding:1.1rem 1rem;
  text-align:center;
}}
.stat:last-child{{border-right:none}}
.stat-n{{font-size:1.6rem;font-weight:700;letter-spacing:-.04em;line-height:1}}
.stat-l{{font-size:.65rem;text-transform:uppercase;letter-spacing:.08em;color:#9a8f7c;margin-top:.3rem}}
.c-ok{{color:#2d6a4f}}.c-fail{{color:#b94040}}

/* ── Section label ── */
.sec-label{{
  font-size:.65rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;
  color:#8a5c00;margin-bottom:1.25rem;padding-bottom:.6rem;
  border-bottom:1px solid #d0c5b0;
}}

/* ── Programs ── */
.program-block{{
  background:#fff;border:1px solid #d0c5b0;border-radius:6px;
  padding:1.25rem 1.5rem;margin-bottom:.75rem;
}}
.program-title{{
  font-size:1.05rem;font-weight:600;letter-spacing:-.02em;
  display:flex;align-items:baseline;gap:.75rem;flex-wrap:wrap;
}}
.domain-tag{{
  font-size:.65rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
  background:rgba(42,77,143,.1);color:#2a4d8f;padding:.2rem .5rem;
  border-radius:3px;flex-shrink:0;
}}
.prio{{font-size:.75rem;color:#9a8f7c;margin-left:auto}}
.question-list{{margin-top:.75rem;padding-top:.75rem;border-top:1px solid #ede5d8}}
.q-item{{font-size:.85rem;color:#5a5044;padding:.3rem 0;padding-left:1rem;border-left:2px solid #d0c5b0;margin-bottom:.35rem}}

/* ── Timeline ── */
.timeline{{position:relative;padding-left:1.75rem}}
.timeline::before{{
  content:'';position:absolute;left:.5rem;top:.75rem;bottom:.75rem;
  width:1px;background:#d0c5b0;
}}
.timeline-item{{position:relative;margin-bottom:2.5rem}}
.tl-dot{{
  position:absolute;left:-1.3rem;top:.55rem;
  width:.7rem;height:.7rem;border-radius:50%;
  background:#d0c5b0;border:2px solid #f4efe6;
  outline:1px solid #d0c5b0;
}}
.tl-body{{background:#fff;border:1px solid #d0c5b0;border-radius:6px;padding:1.25rem 1.5rem}}
.hyp-stmt{{font-size:.97rem;font-weight:600;letter-spacing:-.01em;margin-bottom:.5rem}}
.hyp-conf{{font-size:.78rem;margin-bottom:1rem}}
.conf-high{{color:#2d6a4f}}.conf-mid{{color:#8a5c00}}.conf-low{{color:#b94040}}

/* ── Plan ── */
.plan-block{{
  background:#faf6ef;border:1px solid #e8e0d0;border-radius:4px;
  padding:1rem 1.25rem;margin-bottom:.75rem;
}}
.plan-desc{{font-size:.85rem;font-weight:600;color:#5a5044;margin-bottom:.5rem}}
.plan-steps{{font-size:.8rem;color:#7a6f5e;padding-left:1.25rem;margin-bottom:.75rem}}
.plan-steps li{{margin-bottom:.3rem}}

/* ── Run ── */
.run-row{{border-top:1px solid #e8e0d0;padding-top:.75rem;margin-top:.5rem}}
.run-status{{font-size:.65rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
  padding:.2rem .5rem;border-radius:3px;}}
.ok{{background:rgba(45,106,79,.12);color:#2d6a4f}}
.fail{{background:rgba(185,64,64,.1);color:#b94040}}
.pending{{background:rgba(138,92,0,.1);color:#8a5c00}}
.run-id-sm{{font-size:.7rem;color:#9a8f7c;margin-left:.5rem}}
.run-artifact{{margin-top:.6rem}}
.art-summary{{font-size:.82rem;font-style:italic;color:#7a6f5e;margin-bottom:.4rem}}
.obs-line{{font-size:.8rem;color:#5a5044;padding:.25rem 0;border-bottom:1px solid #f0e8dc}}
.obs-line:last-child{{border-bottom:none}}
details.obs-detail summary{{font-size:.78rem;color:#7a6f5e;cursor:pointer;padding:.25rem 0}}
details.obs-detail pre{{font-size:.72rem;background:#f0e8dc;padding:.75rem;border-radius:3px;white-space:pre-wrap;margin-top:.4rem}}

/* ── Claim chip ── */
.claim-chip{{
  display:inline-block;font-size:.75rem;
  background:rgba(45,106,79,.1);color:#2d6a4f;
  border:1px solid rgba(45,106,79,.25);
  border-radius:3px;padding:.2rem .6rem;margin:.25rem .25rem 0 0;
}}
.claim-chip.conf-mid{{background:rgba(138,92,0,.1);color:#8a5c00;border-color:rgba(138,92,0,.25)}}
.claim-chip.conf-low{{background:rgba(185,64,64,.08);color:#b94040;border-color:rgba(185,64,64,.2)}}
.chip-conf{{opacity:.6;margin-left:.35rem}}

/* ── Claims grid ── */
.claims-grid{{display:grid;grid-template-columns:1fr 1fr;gap:.75rem}}
@media(max-width:640px){{.claims-grid{{grid-template-columns:1fr}}}}
.claim-card{{background:#fff;border:1px solid #d0c5b0;border-radius:6px;padding:1rem 1.25rem}}
.claim-card.conf-high{{border-left:3px solid #2d6a4f}}
.claim-card.conf-mid{{border-left:3px solid #8a5c00}}
.claim-card.conf-low{{border-left:3px solid #b94040}}
.claim-text{{font-size:.88rem;font-weight:500;margin-bottom:.6rem}}
.claim-conf{{display:flex;align-items:center;gap:.6rem}}
.conf-bar{{height:4px;background:#2d6a4f;border-radius:2px;flex-shrink:0;max-width:120px}}
.conf-bar + span{{font-size:.72rem;font-family:monospace;color:#9a8f7c}}

/* ── Empty state ── */
.empty-state{{font-size:.88rem;color:#9a8f7c;font-style:italic;padding:1.5rem 0}}

/* ── Graph JSON ── */
.json-section .sec-label{{display:flex;align-items:center;gap:1rem}}
.toggle-btn{{font-size:.7rem;padding:.2rem .65rem;border:1px solid #d0c5b0;border-radius:3px;background:transparent;cursor:pointer;color:#5a5044}}
.toggle-btn:hover{{background:#e8e0d0}}
.graph-json{{font-size:.7rem;line-height:1.5;background:#1a1712;color:#c8bda8;padding:1.5rem;border-radius:6px;overflow:auto;max-height:500px;white-space:pre}}
.hidden{{display:none}}

/* ── Responsive ── */
@media(max-width:720px){{
  .stats-row{{grid-template-columns:repeat(3,1fr)}}
  .stat:nth-child(3){{border-right:none}}
}}
</style>
</head>
<body>
<nav>
  <a href="#" class="nav-logo">chitta-research</a>
  <div class="nav-links">
    <a href="#top">Summary</a>
    <a href="#hypotheses">Hypotheses</a>
    <a href="#claims">Claims</a>
    <a href="#graph">Graph</a>
  </div>
</nav>
<div class="page">
  <div id="top">{body}</div>
</div>
<script>
function toggleJson(){{
  var el = document.getElementById('graph-json');
  el.classList.toggle('hidden');
}}
</script>
</body>
</html>"##, body = body)
}
