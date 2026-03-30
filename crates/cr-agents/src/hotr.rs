use std::cmp::Ordering;

use async_trait::async_trait;
use cr_llm::{CompletionRequest, Message};
use cr_types::*;

use crate::{Agent, AgentAction, AgentContext};

pub struct Hotr;

const SYSTEM_PROMPT: &str = r#"You are a scientific hypothesis generator. Given a research question and context, generate 2-3 specific, testable hypotheses with concrete experiment plans.

CRITICAL CONSTRAINTS FOR EXPERIMENT PLANS — read carefully before generating steps:

1. ONLY use commands and files that are GUARANTEED TO EXIST. Never reference Python scripts,
   data files, or binaries that would need to be created first. If you need a script, write it
   inline with `python3 -c "..."` or `awk '...'` or `bash -c "..."`.

2. NEVER reference scripts like `scripts/foo.py`, `analysis/bar.R`, or custom binaries.
   These do not exist and will cause immediate failure.

3. For code analysis use ONLY these always-available tools:
   - grep, find, wc, awk, sed, sort, uniq, head, tail
   - python3 -c "..." for inline computation
   - cargo clippy / cargo test / cargo build (with correct --manifest-path)
   - git log, git diff, git shortlog

4. CARGO MANIFEST PATH: for chitta-research crates, always use:
   --manifest-path /maps/projects/fernandezguerra/apps/repos/chitta-research/Cargo.toml

5. Do NOT assume any specific file paths exist unless you know they do. Use find to discover
   paths first if uncertain: `run: find /maps/projects/fernandezguerra/apps/repos/chitta-research -name "*.rs" | head -20`

6. Each `run:` step must be a complete, self-contained shell command that will succeed on its
   own. Never chain steps that depend on artifacts from previous steps.

7. Keep steps simple and observable — prefer 3 working grep/wc steps over 1 failing pipeline.

For steps that run actual commands, prefix with "run:" so the executor dispatches them as subprocesses.

Respond with ONLY a JSON array:
[
  {
    "statement": "Specific testable hypothesis statement",
    "prior_confidence": 0.0-1.0,
    "experiment_plan": {
      "description": "How to test this hypothesis",
      "steps": ["step1", "step2", ...]
    }
  }
]"#;

#[derive(serde::Deserialize)]
struct HypothesisResponse {
    statement: String,
    prior_confidence: f32,
    experiment_plan: ExperimentPlanResponse,
}

#[derive(serde::Deserialize)]
struct ExperimentPlanResponse {
    #[allow(dead_code)]
    description: String,
    steps: Vec<String>,
}

fn cmp_program_priority_desc(a: f32, b: f32) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => b.total_cmp(&a),
    }
}

#[async_trait]
impl Agent for Hotr {
    fn name(&self) -> &str {
        "hotr"
    }

    async fn step(&self, ctx: &AgentContext) -> Result<AgentAction, anyhow::Error> {
        let (prog_title, prog_domain, q_text, q_id) = {
            let graph = ctx.graph.read().await;
            let all_nodes = graph.all_nodes();

            let mut programs: Vec<_> = all_nodes
                .iter()
                .filter_map(|n| match &n.kind {
                    NodeKind::ResearchProgram(p) => Some((n.id, p.clone())),
                    _ => None,
                })
                .collect();
            programs.sort_by(|a, b| cmp_program_priority_desc(a.1.priority, b.1.priority));

            // Allow up to 3 hypothesis rounds per question.
            // Iterate programs in priority order; within each program pick the
            // question with the fewest hypotheses (< MAX) to maximise coverage.
            const MAX_HYPOTHESES_PER_QUESTION: usize = 3;
            let mut chosen = None;
            'outer: for (prog_id, program) in &programs {
                let prog_id = *prog_id;
                let questions: Vec<_> = all_nodes
                    .iter()
                    .filter_map(|n| match &n.kind {
                        NodeKind::Question(q) if q.program_id == prog_id => Some(*n),
                        _ => None,
                    })
                    .collect();
                if questions.is_empty() { continue; }

                let mut best: Option<(usize, usize)> = None; // (idx, n_hyps)
                for (i, q) in questions.iter().enumerate() {
                    let derived = graph.children(q.id, EdgeKind::DerivedFrom);
                    let n_hyps = derived.iter()
                        .filter(|n| matches!(n.kind, NodeKind::Hypothesis(_)))
                        .count();
                    if n_hyps < MAX_HYPOTHESES_PER_QUESTION {
                        if best.map(|(_, bn)| n_hyps < bn).unwrap_or(true) {
                            best = Some((i, n_hyps));
                        }
                    }
                }
                if let Some((idx, _)) = best {
                    let q = &questions[idx];
                    let text = match &q.kind {
                        NodeKind::Question(q) => q.text.clone(),
                        _ => unreachable!(),
                    };
                    chosen = Some((program.title.clone(), program.domain.clone(), text, q.id));
                    break 'outer;
                }
            }

            let Some(result) = chosen else {
                return Ok(AgentAction::Noop);
            };
            result
        };

        // Inject relevant code context from chitta if a codebase is configured.
        // Limit to 2000 chars to avoid bloating the room prompt.
        let (code_context, web_refs) = if let Ok(mut chitta) = ctx.chitta.try_lock() {
            let code = if !ctx.codebase_path.is_empty() {
                chitta.code_context(&q_text).await
                    .ok()
                    .filter(|s| !s.is_empty())
                    .map(|s| format!("\n\nRelevant code context from {}:\n{}", ctx.codebase_path, &s[..s.len().min(2000)]))
                    .unwrap_or_default()
            } else {
                String::new()
            };
            // Recall web research findings stored by the Researcher agent.
            let refs = chitta.recall(&format!("web-research {q_text}"), 6).await
                .ok()
                .map(|hits| {
                    let filtered: Vec<String> = hits.into_iter()
                        .filter(|h| h.kind == "wisdom" && h.content.contains("[web-research:"))
                        .map(|h| h.content)
                        .collect();
                    filtered
                })
                .filter(|v| !v.is_empty())
                .map(|v| format!("\n\nRelevant literature (from web research):\n{}", v.join("\n---\n")))
                .unwrap_or_default();
            (code, refs)
        } else {
            (String::new(), String::new())
        };

        let user_msg = format!(
            "Research program: {prog_title}\nDomain: {prog_domain}\n\nQuestion: {q_text}\
             {code_context}{web_refs}\n\n\
             REQUIRED OUTPUT FORMAT — respond with ONLY this JSON array structure, \
             no other text before or after:\n\
             [\n  {{\n    \"statement\": \"specific testable hypothesis\",\n\
             \"prior_confidence\": 0.65,\n    \"experiment_plan\": {{\n\
             \"description\": \"how to test\",\n      \"steps\": [\"step1\", \"step2\"]\n\
             }}\n  }}\n]"
        );

        let resp = ctx.llm.complete(CompletionRequest {
            model: String::new(),
            system: SYSTEM_PROMPT.to_string(),
            messages: vec![Message { role: "user".into(), content: user_msg }],
            max_tokens: 2048,
            temperature: 0.7,
        }).await?;

        let content = resp.content.trim();
        // Find the first JSON array — look for `[{` to skip prose like "[Note: ...]"
        let json_str = if let Some(start) = content.find("[{") {
            if let Some(end) = content.rfind(']') {
                &content[start..=end]
            } else {
                content
            }
        } else if let Some(start) = content.find('[') {
            if let Some(end) = content.rfind(']') {
                &content[start..=end]
            } else {
                content
            }
        } else {
            content
        };

        let hypotheses: Vec<HypothesisResponse> = serde_json::from_str(json_str)
            .map_err(|e| anyhow::anyhow!("Hotr JSON parse failed: {e}\nraw response (first 300 chars): {}", &content[..content.len().min(300)]))?;

        let mut graph = ctx.graph.write().await;
        let mut last_action = AgentAction::Noop;

        for hyp in hypotheses {
            let h_id = NodeId::new();
            let h_node = TypedNode::new(
                h_id,
                NodeKind::Hypothesis(Hypothesis {
                    statement: hyp.statement.clone(),
                    prior_confidence: hyp.prior_confidence,
                    posterior_confidence: None,
                    generating_model: "llm".into(),
                    tier: 1,
                }),
            );
            graph.add_node(h_node)?;
            graph.add_edge(h_id, q_id, EpistemicEdge {
                kind: EdgeKind::DerivedFrom,
                weight: 1.0,
                evidence_ids: vec![],
            })?;

            let plan_id = NodeId::new();
            let plan_node = TypedNode::new(
                plan_id,
                NodeKind::ExperimentPlan(ExperimentPlan {
                    hypothesis_id: h_id,
                    steps: hyp.experiment_plan.steps,
                    estimated_cost_usd: 0.5,
                }),
            );
            graph.add_node(plan_node)?;
            graph.add_edge(plan_id, h_id, EpistemicEdge {
                kind: EdgeKind::DerivedFrom,
                weight: 1.0,
                evidence_ids: vec![],
            })?;

            tracing::info!(hypothesis = %hyp.statement, "hotr: generated hypothesis");
            last_action = AgentAction::AddNode {
                kind: NodeKind::Hypothesis(Hypothesis {
                    statement: hyp.statement,
                    prior_confidence: hyp.prior_confidence,
                    posterior_confidence: None,
                    generating_model: "llm".into(),
                    tier: 1,
                }),
                edges: vec![(q_id, EpistemicEdge {
                    kind: EdgeKind::DerivedFrom,
                    weight: 1.0,
                    evidence_ids: vec![],
                })],
            };
        }

        drop(graph);

        if let Ok(mut chitta) = ctx.chitta.try_lock() {
            let _ = chitta.remember(
                &format!("Generated hypotheses for question: {}", q_text),
                "research_event",
                &["hotr", "hypothesis"],
                0.7,
            ).await;
        }

        Ok(last_action)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cr_types::ResearchProgram;

    #[test]
    fn cmp_program_priority_desc_sorts_descending_and_places_nan_last() {
        let mut programs = vec![
            ResearchProgram { title: "nan".into(), domain: "test".into(), priority: f32::NAN, max_budget_usd: 1.0 },
            ResearchProgram { title: "high".into(), domain: "test".into(), priority: 2.0, max_budget_usd: 1.0 },
            ResearchProgram { title: "low".into(), domain: "test".into(), priority: 1.0, max_budget_usd: 1.0 },
        ];
        programs.sort_by(|a, b| cmp_program_priority_desc(a.priority, b.priority));
        let titles: Vec<_> = programs.into_iter().map(|p| p.title).collect();
        assert_eq!(titles, vec!["high", "low", "nan"]);
    }
}
