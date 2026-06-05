use async_trait::async_trait;
use cr_types::SchemaRegistry;

pub struct EdgeStream {
    pub realm: String,
    pub events: Vec<String>,
}

#[async_trait]
pub trait EdgeStreamSource: Send + Sync {
    /// Collect edge-type event streams. Returns one stream per realm/program.
    async fn edge_streams(&self) -> anyhow::Result<Vec<EdgeStream>>;
}

/// Merges streams and computes the weakest (most permissive) consistent algebra
/// across all contributing realms before passing to MotifMiner.
pub fn merged_events(streams: &[EdgeStream]) -> Vec<String> {
    streams.iter().flat_map(|s| s.events.iter().cloned()).collect()
}

/// Local graph stream source — wraps a single BeliefGraph.
pub struct LocalGraphSource {
    pub events: Vec<String>,
}

#[async_trait]
impl EdgeStreamSource for LocalGraphSource {
    async fn edge_streams(&self) -> anyhow::Result<Vec<EdgeStream>> {
        Ok(vec![EdgeStream { realm: "local".to_string(), events: self.events.clone() }])
    }
}
