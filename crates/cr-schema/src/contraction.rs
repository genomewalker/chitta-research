
use cr_types::SchemaRegistry;
use crate::morphism::{SchemaMorphism, SchemaOp};

/// Check whether two edge types are distributionally equivalent enough to merge.
/// Uses conditional next-edge-type KL divergence — NOT marginal frequency.
/// Returns a SchemaMorphism(MergeEdgeTypes) if KL < threshold, else None.
pub fn propose_merge(
    type_a: &str,
    type_b: &str,
    events: &[String],
    registry: &SchemaRegistry,
    kl_threshold: f64,
) -> Option<SchemaMorphism> {
    let kl = conditional_context_kl(type_a, type_b, events);
    if kl > kl_threshold { return None; }
    // keep the one with more support (higher usage count)
    let count_a = events.iter().filter(|e| e.as_str() == type_a).count();
    let count_b = events.iter().filter(|e| e.as_str() == type_b).count();
    let (keep, drop) = if count_a >= count_b { (type_a, type_b) } else { (type_b, type_a) };
    Some(SchemaMorphism {
        ops: vec![SchemaOp::MergeEdgeTypes { keep: keep.to_string(), drop: drop.to_string() }],
        rationale: format!("merge {type_a}+{type_b}: conditional KL={kl:.4} < {kl_threshold}"),
    })
}

/// KL divergence on conditional next-edge-type distribution:
/// KL(P(next|a) || P(next|b))
/// Symmetrized as 0.5*(KL(a||b) + KL(b||a))
fn conditional_context_kl(type_a: &str, type_b: &str, events: &[String]) -> f64 {
    use std::collections::HashMap;
    let mut ctx_a: HashMap<&str, usize> = HashMap::new();
    let mut ctx_b: HashMap<&str, usize> = HashMap::new();
    for window in events.windows(2) {
        if window[0] == type_a { *ctx_a.entry(window[1].as_str()).or_insert(0) += 1; }
        if window[0] == type_b { *ctx_b.entry(window[1].as_str()).or_insert(0) += 1; }
    }
    let total_a: f64 = ctx_a.values().sum::<usize>() as f64;
    let total_b: f64 = ctx_b.values().sum::<usize>() as f64;
    if total_a == 0.0 || total_b == 0.0 { return f64::INFINITY; }
    let all_keys: std::collections::HashSet<&str> = ctx_a.keys().chain(ctx_b.keys()).copied().collect();
    let alpha = 0.01; // Laplace smoothing
    let k = all_keys.len() as f64;
    let mut kl_ab = 0.0f64;
    let mut kl_ba = 0.0f64;
    for key in &all_keys {
        let p = (*ctx_a.get(key).unwrap_or(&0) as f64 + alpha) / (total_a + k * alpha);
        let q = (*ctx_b.get(key).unwrap_or(&0) as f64 + alpha) / (total_b + k * alpha);
        kl_ab += p * (p / q).ln();
        kl_ba += q * (q / p).ln();
    }
    0.5 * (kl_ab + kl_ba)
}

/// Propose retirement for types that are inactive in the event stream.
/// A type is stale if it hasn't appeared in the last window events.
pub fn propose_retirements(events: &[String], registry: &SchemaRegistry, window: usize) -> Vec<SchemaMorphism> {
    let recent: std::collections::HashSet<&str> = events[events.len().saturating_sub(window)..].iter().map(|s| s.as_str()).collect();
    registry.custom_edge_specs().into_iter()
        .filter(|spec| spec.lifecycle == cr_types::EdgeLifecycle::Active)
        .filter(|spec| !recent.contains(spec.name.as_str()))
        .map(|spec| SchemaMorphism {
            ops: vec![SchemaOp::RetireEdgeType { name: spec.name.clone() }],
            rationale: format!("retire {}: absent from last {window} events", spec.name),
        })
        .collect()
}
