use cr_types::{EdgeAlgebra, EdgeKind, SchemaRegistry};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchemaOp {
    AddEdgeType { name: String, algebra: EdgeAlgebra },
    AddNodeType { name: String },
    MergeEdgeTypes { keep: String, drop: String },
    RetireEdgeType { name: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaMorphism {
    pub ops: Vec<SchemaOp>,
    pub rationale: String,
}

#[derive(Debug, Error)]
pub enum MigrateError {
    #[error("schema op is invalid: {0}")]
    InvalidOp(String),
    #[error("merge target '{0}' not found in registry")]
    UnknownType(String),
}

pub fn migrate(m: &SchemaMorphism, registry: &SchemaRegistry) -> Result<SchemaRegistry, MigrateError> {
    let mut r = registry.clone();
    for op in &m.ops {
        match op {
            SchemaOp::AddEdgeType { name, algebra } => {
                r.register(name.clone(), algebra.clone());
            }
            SchemaOp::AddNodeType { name: _ } => {}
            SchemaOp::MergeEdgeTypes { keep, drop } => {
                if r.get(&EdgeKind::Custom(keep.clone())).is_none() {
                    return Err(MigrateError::UnknownType(keep.clone()));
                }
                r.remove(drop);
            }
            SchemaOp::RetireEdgeType { name } => {
                r.remove(name);
            }
        }
    }
    Ok(r)
}
