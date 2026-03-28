use cr_chitta::ChittaClient;
use cr_graph::BeliefGraph;
use cr_types::*;

#[derive(Debug)]
pub struct ReconsolidationReport {
    pub corrected_node: NodeId,
    pub affected_nodes: Vec<NodeId>,
    pub confidence_reductions: Vec<(NodeId, f32, f32)>, // id, old, new
}

pub async fn trigger_reconsolidation(
    graph: &mut BeliefGraph,
    corrected_node: NodeId,
    chitta: Option<&mut ChittaClient>,
) -> Result<ReconsolidationReport, anyhow::Error> {
    let downstream = graph.descendants(corrected_node, EdgeKind::DerivedFrom);

    let mut confidence_reductions = Vec::new();

    for node_id in &downstream {
        let Some(node) = graph.get_node_mut(*node_id) else {
            continue;
        };

        match &mut node.kind {
            NodeKind::Hypothesis(h) if h.tier >= 2 => {
                if let Some(post) = h.posterior_confidence.as_mut() {
                    let old = *post;
                    *post = (*post - 0.2).max(0.0);
                    confidence_reductions.push((*node_id, old, *post));
                }
            }
            NodeKind::Claim(c) => {
                let old = c.confidence;
                c.confidence = (c.confidence - 0.2).max(0.0);
                confidence_reductions.push((*node_id, old, c.confidence));
            }
            _ => {}
        }
    }

    // Add warning edges from corrected node to affected nodes
    for node_id in &downstream {
        let _ = graph.add_edge(
            corrected_node,
            *node_id,
            EpistemicEdge {
                kind: EdgeKind::BlockedBy,
                weight: 0.5,
                evidence_ids: vec![corrected_node],
            },
        );
    }

    if let Some(chitta) = chitta {
        let summary = format!(
            "Reconsolidation triggered for node {}. {} downstream nodes affected, {} confidence reductions applied.",
            corrected_node, downstream.len(), confidence_reductions.len()
        );
        let _ = chitta.observe("reconsolidation", "reconsolidation_event", &summary).await;
    }

    tracing::info!(
        corrected = %corrected_node,
        affected = downstream.len(),
        reductions = confidence_reductions.len(),
        "reconsolidation complete"
    );

    Ok(ReconsolidationReport {
        corrected_node,
        affected_nodes: downstream,
        confidence_reductions,
    })
}
