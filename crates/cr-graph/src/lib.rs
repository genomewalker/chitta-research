use std::collections::HashMap;

use petgraph::stable_graph::{NodeIndex, StableDiGraph};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use serde::{Deserialize, Serialize};

use cr_types::{
    EdgeKind, EpistemicEdge, NodeId, NodeKind, TypedNode,
};

#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("node not found: {0}")]
    NodeNotFound(NodeId),
    #[error("duplicate node: {0}")]
    DuplicateNode(NodeId),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

pub struct BeliefGraph {
    graph: StableDiGraph<TypedNode, EpistemicEdge>,
    index: HashMap<NodeId, NodeIndex>,
}

impl BeliefGraph {
    pub fn new() -> Self {
        Self {
            graph: StableDiGraph::new(),
            index: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node: TypedNode) -> Result<NodeId, GraphError> {
        if self.index.contains_key(&node.id) {
            return Err(GraphError::DuplicateNode(node.id));
        }
        let id = node.id;
        let idx = self.graph.add_node(node);
        self.index.insert(id, idx);
        Ok(id)
    }

    pub fn add_edge(
        &mut self,
        from: NodeId,
        to: NodeId,
        edge: EpistemicEdge,
    ) -> Result<(), GraphError> {
        let from_idx = *self.index.get(&from).ok_or(GraphError::NodeNotFound(from))?;
        let to_idx = *self.index.get(&to).ok_or(GraphError::NodeNotFound(to))?;
        self.graph.add_edge(from_idx, to_idx, edge);
        Ok(())
    }

    pub fn remove_node(&mut self, id: NodeId) -> Result<TypedNode, GraphError> {
        let idx = *self.index.get(&id).ok_or(GraphError::NodeNotFound(id))?;
        let node = self.graph.remove_node(idx).ok_or(GraphError::NodeNotFound(id))?;
        self.index.remove(&id);
        Ok(node)
    }

    pub fn get_node(&self, id: NodeId) -> Option<&TypedNode> {
        self.index.get(&id).and_then(|idx| self.graph.node_weight(*idx))
    }

    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut TypedNode> {
        self.index.get(&id).copied().and_then(|idx| self.graph.node_weight_mut(idx))
    }

    pub fn all_nodes(&self) -> Vec<&TypedNode> {
        self.graph.node_indices().filter_map(|idx| self.graph.node_weight(idx)).collect()
    }

    pub fn children(&self, id: NodeId, edge_kind: EdgeKind) -> Vec<&TypedNode> {
        self.edges_by_kind(id, edge_kind, Direction::Incoming)
    }

    pub fn parents(&self, id: NodeId, edge_kind: EdgeKind) -> Vec<&TypedNode> {
        self.edges_by_kind(id, edge_kind, Direction::Outgoing)
    }

    pub fn descendants(&self, id: NodeId, edge_kind: EdgeKind) -> Vec<NodeId> {
        let mut result = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        let mut visited = std::collections::HashSet::new();
        queue.push_back(id);
        visited.insert(id);
        while let Some(current) = queue.pop_front() {
            for child in self.children(current, edge_kind) {
                if visited.insert(child.id) {
                    result.push(child.id);
                    queue.push_back(child.id);
                }
            }
        }
        result
    }

    pub fn hypotheses_for_program(&self, program_id: NodeId) -> Vec<&TypedNode> {
        let Some(&prog_idx) = self.index.get(&program_id) else {
            return Vec::new();
        };
        self.graph
            .neighbors_directed(prog_idx, Direction::Incoming)
            .filter_map(|idx| {
                let node = self.graph.node_weight(idx)?;
                matches!(node.kind, NodeKind::Hypothesis(_)).then_some(node)
            })
            .collect()
    }

    pub fn supporting_evidence(&self, node_id: NodeId) -> Vec<&TypedNode> {
        self.edges_by_kind(node_id, EdgeKind::Supports, Direction::Incoming)
    }

    pub fn refuting_evidence(&self, node_id: NodeId) -> Vec<&TypedNode> {
        self.edges_by_kind(node_id, EdgeKind::Refutes, Direction::Incoming)
    }

    fn edges_by_kind(
        &self,
        node_id: NodeId,
        kind: EdgeKind,
        direction: Direction,
    ) -> Vec<&TypedNode> {
        let Some(&idx) = self.index.get(&node_id) else {
            return Vec::new();
        };
        self.graph
            .edges_directed(idx, direction)
            .filter(|e| e.weight().kind == kind)
            .filter_map(|e| {
                let other = match direction {
                    Direction::Incoming => e.source(),
                    Direction::Outgoing => e.target(),
                };
                self.graph.node_weight(other)
            })
            .collect()
    }

    pub fn pareto_frontier(&self) -> Vec<NodeId> {
        let with_fitness: Vec<_> = self
            .graph
            .node_indices()
            .filter_map(|idx| {
                let node = self.graph.node_weight(idx)?;
                Some((node.id, node.fitness?))
            })
            .collect();

        cr_fitness::pareto_frontier_from_pairs(&with_fitness)
    }

    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    pub fn snapshot_to_json(&self) -> Result<String, GraphError> {
        let snapshot = GraphSnapshot::from_graph(self);
        Ok(serde_json::to_string_pretty(&snapshot)?)
    }

    pub fn from_json_snapshot(json: &str) -> Result<Self, GraphError> {
        let snapshot: GraphSnapshot = serde_json::from_str(json)?;
        Ok(snapshot.into_graph())
    }
}

impl Default for BeliefGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Serialize, Deserialize)]
struct GraphSnapshot {
    nodes: Vec<TypedNode>,
    edges: Vec<SnapshotEdge>,
}

#[derive(Serialize, Deserialize)]
struct SnapshotEdge {
    from: NodeId,
    to: NodeId,
    edge: EpistemicEdge,
}

impl GraphSnapshot {
    fn from_graph(g: &BeliefGraph) -> Self {
        let nodes: Vec<_> = g
            .graph
            .node_indices()
            .filter_map(|idx| g.graph.node_weight(idx).cloned())
            .collect();

        let edges: Vec<_> = g
            .graph
            .edge_indices()
            .filter_map(|idx| {
                let (src, dst) = g.graph.edge_endpoints(idx)?;
                let from_node = g.graph.node_weight(src)?;
                let to_node = g.graph.node_weight(dst)?;
                let edge = g.graph.edge_weight(idx)?.clone();
                Some(SnapshotEdge {
                    from: from_node.id,
                    to: to_node.id,
                    edge,
                })
            })
            .collect();

        Self { nodes, edges }
    }

    fn into_graph(self) -> BeliefGraph {
        let mut g = BeliefGraph::new();
        for node in self.nodes {
            let _ = g.add_node(node);
        }
        for se in self.edges {
            let _ = g.add_edge(se.from, se.to, se.edge);
        }
        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cr_types::*;

    fn make_hypothesis(statement: &str) -> TypedNode {
        TypedNode::new(
            NodeId::new(),
            NodeKind::Hypothesis(Hypothesis {
                statement: statement.into(),
                prior_confidence: 0.5,
                posterior_confidence: None,
                generating_model: "test".into(),
                tier: 1,
            }),
        )
    }

    fn make_program(title: &str) -> TypedNode {
        TypedNode::new(
            NodeId::new(),
            NodeKind::ResearchProgram(ResearchProgram {
                title: title.into(),
                domain: "test".into(),
                priority: 1.0,
                max_budget_usd: 10.0,
            }),
        )
    }

    #[test]
    fn add_and_retrieve_nodes() {
        let mut g = BeliefGraph::new();
        let node = make_hypothesis("test hypothesis");
        let id = node.id;
        g.add_node(node).unwrap();
        assert!(g.get_node(id).is_some());
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn duplicate_node_rejected() {
        let mut g = BeliefGraph::new();
        let node = make_hypothesis("h1");
        let id = node.id;
        g.add_node(node.clone()).unwrap();
        assert!(matches!(
            g.add_node(node),
            Err(GraphError::DuplicateNode(dup_id)) if dup_id == id
        ));
    }

    #[test]
    fn add_edge_and_query_evidence() {
        let mut g = BeliefGraph::new();
        let h = make_hypothesis("h1");
        let h_id = h.id;
        let obs = TypedNode::new(
            NodeId::new(),
            NodeKind::Observation(Observation {
                run_id: NodeId::new(),
                summary: "observed X".into(),
                data_ref: None,
            }),
        );
        let obs_id = obs.id;
        g.add_node(h).unwrap();
        g.add_node(obs).unwrap();
        g.add_edge(
            obs_id,
            h_id,
            EpistemicEdge {
                kind: EdgeKind::Supports,
                weight: 0.8,
                evidence_ids: vec![],
            },
        )
        .unwrap();
        let support = g.supporting_evidence(h_id);
        assert_eq!(support.len(), 1);
        assert_eq!(support[0].id, obs_id);
        assert!(g.refuting_evidence(h_id).is_empty());
    }

    #[test]
    fn remove_node() {
        let mut g = BeliefGraph::new();
        let node = make_hypothesis("to remove");
        let id = node.id;
        g.add_node(node).unwrap();
        g.remove_node(id).unwrap();
        assert!(g.get_node(id).is_none());
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn hypotheses_for_program() {
        let mut g = BeliefGraph::new();
        let prog = make_program("ARG research");
        let prog_id = prog.id;
        let h = make_hypothesis("ARGs increase with depth");
        let h_id = h.id;
        g.add_node(prog).unwrap();
        g.add_node(h).unwrap();
        g.add_edge(
            h_id,
            prog_id,
            EpistemicEdge {
                kind: EdgeKind::DerivedFrom,
                weight: 1.0,
                evidence_ids: vec![],
            },
        )
        .unwrap();
        let hyps = g.hypotheses_for_program(prog_id);
        assert_eq!(hyps.len(), 1);
    }

    #[test]
    fn snapshot_roundtrip() {
        let mut g = BeliefGraph::new();
        let h1 = make_hypothesis("h1");
        let h2 = make_hypothesis("h2");
        let h1_id = h1.id;
        let h2_id = h2.id;
        g.add_node(h1).unwrap();
        g.add_node(h2).unwrap();
        g.add_edge(
            h1_id,
            h2_id,
            EpistemicEdge {
                kind: EdgeKind::Supports,
                weight: 0.9,
                evidence_ids: vec![],
            },
        )
        .unwrap();
        let json = g.snapshot_to_json().unwrap();
        let restored = BeliefGraph::from_json_snapshot(&json).unwrap();
        assert_eq!(restored.node_count(), 2);
        assert_eq!(restored.edge_count(), 1);
    }
}
