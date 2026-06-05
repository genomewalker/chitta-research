use cr_schema::{MotifMiner, Proposer, SchemaGate, SchemaOp, SwitchKT};
use cr_types::{FitnessCue, NodeId, SchemaRegistry};

#[test]
fn motif_recovery() {
    let events: Vec<String> = (0..20)
        .flat_map(|_| vec!["Supports".to_string(), "DerivedFrom".to_string()])
        .collect();
    let schema = SchemaRegistry::new();
    let cue = FitnessCue { run_id: NodeId::new(), novelty: 0.9, calibration_improvement: 0.1 };
    let proposals = MotifMiner { min_support: 3 }.propose(&events, &schema, &cue);
    assert!(!proposals.is_empty(), "should propose Supports→DerivedFrom motif");
    let names: Vec<_> = proposals.iter()
        .flat_map(|m| m.ops.iter())
        .filter_map(|op| if let SchemaOp::AddEdgeType { name, .. } = op { Some(name.clone()) } else { None })
        .collect();
    assert!(names.iter().any(|n| n.contains("Supports") && n.contains("DerivedFrom")));
}

/// A stream too short to cross min_support must produce zero proposals and zero accepted types.
#[test]
fn null_control_no_false_discoveries() {
    // 7 events — no bigram can appear >= 10 times, so MotifMiner produces no proposals.
    let types = ["Supports", "Refutes", "DerivedFrom", "GeneralizesTo", "BlockedBy"];
    let events: Vec<String> = (0..7).map(|i| types[i % 5].to_string()).collect();
    let schema = SchemaRegistry::new();
    let mut gate = SchemaGate::new(SwitchKT::default());
    let cue = FitnessCue { run_id: NodeId::new(), novelty: 0.9, calibration_improvement: 0.1 };
    let proposals = MotifMiner { min_support: 10 }.propose(&events, &schema, &cue);
    assert!(proposals.is_empty(), "stream too short to mine motifs should produce zero proposals");
    let accepted: Vec<_> = proposals.into_iter()
        .map(|m| gate.evaluate(m, &events, &schema))
        .filter(|v| v.accepted)
        .collect();
    assert!(accepted.is_empty(), "zero proposals means zero accepted types");
}

/// A morphism placed in the rejection ledger must be accepted when more data makes it worthwhile.
#[test]
fn resurrection_at_scale() {
    use cr_schema::morphism::{SchemaOp, SchemaMorphism};
    use cr_types::{EdgeAlgebra, EdgeSign};

    let schema = SchemaRegistry::new();
    let mut gate = SchemaGate::new(SwitchKT::default());

    // Seed the ledger with a previously-rejected morphism (delta_bits was 0.5 > 0).
    let m = SchemaMorphism {
        ops: vec![SchemaOp::AddEdgeType {
            name: "motif:Supports_DerivedFrom".to_string(),
            algebra: EdgeAlgebra {
                acyclic: false, transitive: false, symmetric: false,
                sign: EdgeSign::None, contradicts: None,
            },
        }],
        rationale: "test".to_string(),
    };
    gate.ledger.record(m, 0.5);
    assert_eq!(gate.ledger.entries().len(), 1);

    // Large biased stream — resweep should now accept the motif.
    let large: Vec<String> = (0..300)
        .flat_map(|_| vec!["Supports".to_string(), "DerivedFrom".to_string()])
        .collect();
    let resurrected = gate.resweep(&large, &schema);
    assert!(!resurrected.is_empty(), "seeded morphism should be resurrected on large biased stream");
    assert_eq!(gate.ledger.entries().len(), 0, "ledger should be empty after resurrection");
}
