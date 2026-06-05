pub mod code;
pub mod morphism;
pub mod proposer;
pub mod gate;
pub mod contraction;
pub mod stream;

pub use code::{MdlCode, SwitchKT};
pub use morphism::{MigrateError, SchemaMorphism, SchemaOp, migrate};
pub use proposer::{make_edge_type_spec, MotifMiner, Proposer, RoleClusterer};
pub use gate::{RejectionLedger, SchemaGate, Verdict};
pub use contraction::{propose_merge, propose_retirements};
pub use stream::{EdgeStream, EdgeStreamSource, LocalGraphSource, merged_events};
