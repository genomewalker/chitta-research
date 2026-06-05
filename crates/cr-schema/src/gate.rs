use crate::code::MdlCode;
use crate::morphism::{SchemaOp, SchemaMorphism, migrate};
use cr_types::SchemaRegistry;
use serde::{Deserialize, Serialize};

/// Replace every occurrence of a motif bigram (A, B) in the event stream with the
/// new type token "motif:A_B". This is the structural re-encoding step that lets the
/// MDL code measure actual compression rather than just schema vocabulary size.
fn apply_morphism_to_stream(m: &SchemaMorphism, events: &[String]) -> Vec<String> {
    // Collect motif bigrams this morphism introduces
    let bigrams: Vec<(String, String, String)> = m.ops.iter().filter_map(|op| {
        if let SchemaOp::AddEdgeType { name, .. } = op {
            if let Some(rest) = name.strip_prefix("motif:") {
                if let Some(idx) = rest.find('_') {
                    return Some((rest[..idx].to_string(), rest[idx+1..].to_string(), name.clone()));
                }
            }
        }
        None
    }).collect();

    if bigrams.is_empty() { return events.to_vec(); }

    let mut out = Vec::with_capacity(events.len());
    let mut i = 0;
    while i < events.len() {
        let mut matched = false;
        if i + 1 < events.len() {
            for (a, b, new_name) in &bigrams {
                if events[i] == *a && events[i + 1] == *b {
                    out.push(new_name.clone());
                    i += 2;
                    matched = true;
                    break;
                }
            }
        }
        if !matched {
            out.push(events[i].clone());
            i += 1;
        }
    }
    out
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Verdict {
    pub morphism: SchemaMorphism,
    pub delta_bits: f64,
    pub accepted: bool,
}

#[derive(Debug, Default, Clone)]
pub struct RejectionLedger {
    entries: Vec<(SchemaMorphism, f64)>,
}

impl RejectionLedger {
    pub fn record(&mut self, m: SchemaMorphism, delta_bits: f64) {
        self.entries.push((m, delta_bits));
    }
    pub fn entries(&self) -> &[(SchemaMorphism, f64)] { &self.entries }
    pub fn remove(&mut self, idx: usize) { self.entries.remove(idx); }
}

pub struct SchemaGate<C: MdlCode> {
    pub code: C,
    pub ledger: RejectionLedger,
}

impl<C: MdlCode> SchemaGate<C> {
    pub fn new(code: C) -> Self {
        Self { code, ledger: RejectionLedger::default() }
    }

    pub fn evaluate(&mut self, m: SchemaMorphism, events: &[String], schema: &SchemaRegistry) -> Verdict {
        let l_before = self.code.length(events, schema);
        let new_schema = match migrate(&m, schema) {
            Ok(s) => s,
            Err(_) => return Verdict { morphism: m, delta_bits: f64::INFINITY, accepted: false },
        };
        // Re-encode the event stream under the new schema: replace relabeled bigrams with the
        // new type token so the MDL code measures actual structural compression.
        let migrated_events = apply_morphism_to_stream(&m, events);
        let l_after = self.code.length(&migrated_events, &new_schema);
        let delta_bits = l_after - l_before + self.code.switch_cost();
        let accepted = delta_bits < 0.0;
        if !accepted {
            self.ledger.record(m.clone(), delta_bits);
        }
        Verdict { morphism: m, delta_bits, accepted }
    }

    /// Re-score rejected proposals with current data (resurrection pass).
    pub fn resweep(&mut self, events: &[String], schema: &SchemaRegistry) -> Vec<Verdict> {
        let mut resurface: Vec<(usize, SchemaMorphism, f64)> = vec![];
        for (i, (m, _)) in self.ledger.entries().iter().cloned().enumerate() {
            let l_before = self.code.length(events, schema);
            let Ok(new_schema) = migrate(&m, schema) else { continue };
            let migrated_events = apply_morphism_to_stream(&m, events);
            let l_after = self.code.length(&migrated_events, &new_schema);
            let delta_bits = l_after - l_before + self.code.switch_cost();
            if delta_bits < 0.0 {
                resurface.push((i, m, delta_bits));
            }
        }
        let mut result = vec![];
        for (i, m, delta_bits) in resurface.into_iter().rev() {
            self.ledger.remove(i);
            result.push(Verdict { morphism: m, delta_bits, accepted: true });
        }
        result
    }
}
