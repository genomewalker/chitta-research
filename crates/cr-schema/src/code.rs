use cr_types::SchemaRegistry;

pub trait MdlCode: Send + Sync {
    fn length(&self, events: &[String], schema: &SchemaRegistry) -> f64;
    fn switch_cost(&self) -> f64;
}

/// KT estimator with forgetting factor gamma. gamma=1.0 is stationary KT;
/// smaller gamma discounts old observations so genuine regime shifts cost a
/// bounded switch_cost rather than inflating L forever.
pub struct SwitchKT {
    pub gamma: f64,
    pub alpha: f64,
}

impl Default for SwitchKT {
    fn default() -> Self {
        Self { gamma: 0.95, alpha: 0.5 }
    }
}

impl MdlCode for SwitchKT {
    fn length(&self, events: &[String], _schema: &SchemaRegistry) -> f64 {
        use std::collections::HashMap;
        let k = events.iter().collect::<std::collections::HashSet<_>>().len().max(1) as f64;
        let mut counts: HashMap<&str, f64> = HashMap::new();
        let mut total: f64 = 0.0;
        let mut length: f64 = 0.0;
        for event in events {
            let n_t = *counts.get(event.as_str()).unwrap_or(&0.0);
            let p = (n_t + self.alpha) / (total + k * self.alpha);
            length -= p.log2();
            for v in counts.values_mut() {
                *v *= self.gamma;
            }
            total *= self.gamma;
            *counts.entry(event.as_str()).or_insert(0.0) += 1.0;
            total += 1.0;
        }
        length
    }

    fn switch_cost(&self) -> f64 {
        (-self.gamma.log2()).abs()
    }
}
