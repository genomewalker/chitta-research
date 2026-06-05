use cr_types::{EdgeAlgebra, EdgeSign, FitnessCue, SchemaRegistry};
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
                        sign: EdgeSign::None, contradicts: None,
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
