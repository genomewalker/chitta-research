use cr_types::{AlgebraEvidence, EdgeAlgebra, EdgeKind, EdgeSign, EdgeTypeSpec, FitnessCue, SchemaRegistry};
use crate::morphism::{SchemaOp, SchemaMorphism};

pub trait Proposer: Send + Sync {
    fn propose(&self, events: &[String], schema: &SchemaRegistry, cue: &FitnessCue) -> Vec<SchemaMorphism>;
}

/// Finds recurring consecutive edge-type pairs as candidate new edge types.
pub struct MotifMiner {
    pub min_support: usize,
}

impl Default for MotifMiner {
    fn default() -> Self { Self { min_support: 3 } }
}

impl Proposer for MotifMiner {
    fn propose(&self, events: &[String], _schema: &SchemaRegistry, _cue: &FitnessCue) -> Vec<SchemaMorphism> {
        use std::collections::HashMap;
        let mut bigrams: HashMap<(String, String), usize> = HashMap::new();
        for window in events.windows(2) {
            *bigrams.entry((window[0].clone(), window[1].clone())).or_insert(0) += 1;
        }
        bigrams.into_iter()
            .filter(|(_, count)| *count >= self.min_support)
            .map(|((a, b), _)| SchemaMorphism {
                ops: vec![SchemaOp::AddEdgeType {
                    name: format!("motif:{a}_{b}"),
                    algebra: EdgeAlgebra {
                        acyclic: false, transitive: false, symmetric: false,
                        sign: EdgeSign::None, contradicts: None, rollback_count: 0,
                    },
                }],
                rationale: format!("Motif {a}→{b} recurs >= {} times", self.min_support),
            })
            .collect()
    }
}

/// Role-based node-type discoverer (SBM-style). Requires graph access — stub until
/// cr-graph integration is added.
pub struct RoleClusterer {
    pub k_max: usize,
}

impl Default for RoleClusterer {
    fn default() -> Self { Self { k_max: 4 } }
}

impl Proposer for RoleClusterer {
    fn propose(&self, _events: &[String], _schema: &SchemaRegistry, _cue: &FitnessCue) -> Vec<SchemaMorphism> {
        vec![]
    }
}

use cr_graph::BeliefGraph;

const MIN_ALGEBRA_SUPPORT: usize = 5;

/// Measure algebraic properties of a motif from actual graph instances.
/// Conservative: only asserts restrictive values (acyclic=true, symmetric=true)
/// when evidence strongly supports them AND support >= MIN_ALGEBRA_SUPPORT.
/// Falls back to permissive (all false) on thin data.
pub fn empirical_algebra(motif_name: &str, graph: &BeliefGraph) -> (EdgeAlgebra, AlgebraEvidence) {
    // Collect all edges of this custom type
    let custom_kind = EdgeKind::Custom(motif_name.to_string());
    let all_edges: Vec<(cr_types::NodeId, cr_types::NodeId)> = graph.all_nodes().iter()
        .flat_map(|n| {
            graph.parents(n.id, custom_kind.clone()).iter()
                .map(|p| (p.id, n.id))
                .collect::<Vec<_>>()
        })
        .collect();
    let support = all_edges.len();
    if support < MIN_ALGEBRA_SUPPORT {
        return (
            EdgeAlgebra { acyclic: false, transitive: false, symmetric: false, sign: EdgeSign::None, contradicts: None, rollback_count: 0 },
            AlgebraEvidence { support, cycle_observed: false, sym_ratio: 0.0, trans_closure: 0.0, confidence: 0.0 },
        );
    }
    // Cycle check via DFS on just these edges
    use std::collections::{HashMap, HashSet};
    let mut adj: HashMap<cr_types::NodeId, Vec<cr_types::NodeId>> = HashMap::new();
    for (from, to) in &all_edges {
        adj.entry(*from).or_default().push(*to);
    }
    let mut cycle_observed = false;
    let mut visited = HashSet::new();
    let mut stack = HashSet::new();
    fn dfs(node: cr_types::NodeId, adj: &HashMap<cr_types::NodeId, Vec<cr_types::NodeId>>, visited: &mut HashSet<cr_types::NodeId>, stack: &mut HashSet<cr_types::NodeId>) -> bool {
        visited.insert(node);
        stack.insert(node);
        if let Some(neighbors) = adj.get(&node) {
            for &nb in neighbors {
                if !visited.contains(&nb) {
                    if dfs(nb, adj, visited, stack) { return true; }
                } else if stack.contains(&nb) {
                    return true;
                }
            }
        }
        stack.remove(&node);
        false
    }
    for &start in adj.keys() {
        if !visited.contains(&start) {
            if dfs(start, &adj, &mut visited, &mut stack) {
                cycle_observed = true;
                break;
            }
        }
    }
    // Symmetry ratio: count (A→B, B→A) pairs vs total
    let edge_set: HashSet<(cr_types::NodeId, cr_types::NodeId)> = all_edges.iter().cloned().collect();
    let sym_pairs = all_edges.iter().filter(|(a, b)| edge_set.contains(&(*b, *a))).count();
    let sym_ratio = if support > 0 { sym_pairs as f32 / support as f32 } else { 0.0 };
    // Transitive closure completeness: for each (A→B, B→C), check if A→C exists
    let mut trans_total = 0usize;
    let mut trans_present = 0usize;
    for (a, b) in &all_edges {
        if let Some(b_neighbors) = adj.get(b) {
            for c in b_neighbors {
                trans_total += 1;
                if edge_set.contains(&(*a, *c)) { trans_present += 1; }
            }
        }
    }
    let trans_closure = if trans_total > 0 { trans_present as f32 / trans_total as f32 } else { 1.0 };
    let confidence = (support as f64 / (support as f64 + 10.0)).min(1.0);
    let algebra = EdgeAlgebra {
        acyclic:     support >= MIN_ALGEBRA_SUPPORT && !cycle_observed,
        transitive:  support >= MIN_ALGEBRA_SUPPORT && trans_closure > 0.95,
        symmetric:   (0.8..=1.2).contains(&sym_ratio),
        sign: EdgeSign::None,
        contradicts: None,
        rollback_count: 0,
    };
    (algebra, AlgebraEvidence { support, cycle_observed, sym_ratio, trans_closure, confidence })
}

/// Create an EdgeTypeSpec from empirical measurement.
/// The enforced algebra is the measured one; declared starts permissive.
pub fn make_edge_type_spec(name: String, graph: &BeliefGraph) -> EdgeTypeSpec {
    let (algebra, evidence) = empirical_algebra(&name, graph);
    let permissive = EdgeAlgebra { acyclic: false, transitive: false, symmetric: false, sign: EdgeSign::None, contradicts: None, rollback_count: 0 };
    EdgeTypeSpec {
        name,
        declared: permissive,
        enforced: algebra,
        evidence,
        lifecycle: cr_types::EdgeLifecycle::Active,
    }
}
