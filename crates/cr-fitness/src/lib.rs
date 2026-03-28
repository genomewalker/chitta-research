use cr_types::{FitnessVector, NodeId};

pub fn pareto_frontier(nodes: &[(&cr_types::TypedNode, FitnessVector)]) -> Vec<NodeId> {
    let pairs: Vec<(NodeId, FitnessVector)> = nodes.iter().map(|(n, f)| (n.id, *f)).collect();
    pareto_frontier_from_pairs(&pairs)
}

pub fn pareto_frontier_from_pairs(pairs: &[(NodeId, FitnessVector)]) -> Vec<NodeId> {
    let mut frontier = Vec::new();
    for (i, (id_a, fit_a)) in pairs.iter().enumerate() {
        let dominated = pairs.iter().enumerate().any(|(j, (_, fit_b))| {
            i != j && fit_b.dominates(fit_a)
        });
        if !dominated {
            frontier.push(*id_a);
        }
    }
    frontier
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fv(novelty: f32, empirical: f32) -> FitnessVector {
        FitnessVector {
            novelty,
            empirical_gain: empirical,
            reproducibility: 0.5,
            cost_efficiency: 0.5,
            transfer_potential: 0.5,
            calibration_improvement: 0.5,
        }
    }

    #[test]
    fn pareto_single_node() {
        let id = NodeId::new();
        let result = pareto_frontier_from_pairs(&[(id, fv(0.5, 0.5))]);
        assert_eq!(result, vec![id]);
    }

    #[test]
    fn pareto_one_dominates() {
        let a = NodeId::new();
        let b = NodeId::new();
        let pairs = vec![
            (a, fv(0.9, 0.9)),
            (b, fv(0.1, 0.1)),
        ];
        let result = pareto_frontier_from_pairs(&pairs);
        assert_eq!(result, vec![a]);
    }

    #[test]
    fn pareto_tradeoff_both_survive() {
        let a = NodeId::new();
        let b = NodeId::new();
        let pairs = vec![
            (a, fv(0.9, 0.1)),
            (b, fv(0.1, 0.9)),
        ];
        let result = pareto_frontier_from_pairs(&pairs);
        assert_eq!(result.len(), 2);
        assert!(result.contains(&a));
        assert!(result.contains(&b));
    }

    #[test]
    fn pareto_empty() {
        let result: Vec<NodeId> = pareto_frontier_from_pairs(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn pareto_three_nodes_mixed() {
        let a = NodeId::new();
        let b = NodeId::new();
        let c = NodeId::new();
        let pairs = vec![
            (a, fv(0.9, 0.9)),
            (b, fv(0.5, 0.5)),
            (c, fv(0.1, 0.95)),
        ];
        let result = pareto_frontier_from_pairs(&pairs);
        assert_eq!(result.len(), 2);
        assert!(result.contains(&a));
        assert!(result.contains(&c));
    }
}
