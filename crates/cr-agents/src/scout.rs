/// Scout agent — natural resource discovery at startup.
///
/// Runs once per session, probes available compute (local GPUs, Slurm partitions,
/// SSH nodes) and stores the topology in chitta as a `kind: wisdom` memory.
/// Adhvaryu can then recall this topology when routing `run:` experiment steps
/// to choose between local execution, `sbatch`, or SSH.
///
/// Named Scout (from the system design: Brahman spawns Scouts for unexplored regions).

use async_trait::async_trait;
use cr_types::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::{Agent, AgentAction, AgentContext};

pub struct Scout {
    script_path: PathBuf,
    done: AtomicBool,
}

impl Scout {
    pub fn new() -> Self {
        let candidates = [
            PathBuf::from("scripts/discover_resources.py"),
            PathBuf::from("/maps/projects/fernandezguerra/apps/repos/chitta-research/scripts/discover_resources.py"),
        ];
        let script_path = candidates.into_iter()
            .find(|p| p.exists())
            .unwrap_or_else(|| PathBuf::from("scripts/discover_resources.py"));
        Self {
            script_path,
            done: AtomicBool::new(false),
        }
    }
}

#[async_trait]
impl Agent for Scout {
    fn name(&self) -> &str { "scout" }

    async fn step(&self, ctx: &AgentContext) -> Result<AgentAction, anyhow::Error> {
        // Run once per session only
        if self.done.load(Ordering::Relaxed) {
            return Ok(AgentAction::Noop);
        }
        self.done.store(true, Ordering::Relaxed);

        if !self.script_path.exists() {
            tracing::warn!(path = %self.script_path.display(), "scout: discover_resources.py not found");
            return Ok(AgentAction::Noop);
        }

        tracing::info!("scout: discovering available compute resources");

        let output = tokio::process::Command::new("python3")
            .arg(&self.script_path)
            .output()
            .await?;

        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr);
            tracing::warn!(error = %err, "scout: resource discovery failed");
            return Ok(AgentAction::Noop);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let topology: serde_json::Value = serde_json::from_str(&stdout)
            .unwrap_or(serde_json::json!({}));

        let summary = &topology["summary"];
        let local = &topology["local"];
        let slurm = &topology["slurm"];

        // Build a human-readable summary for chitta recall
        let mut lines = vec![
            format!("[resource-topology] Compute environment discovered at session start"),
        ];

        let host = local["hostname"].as_str().unwrap_or("unknown");
        let cpus = local["cpu_cores"].as_u64().unwrap_or(0);
        let ram  = local["ram_gb"].as_f64().unwrap_or(0.0);
        let local_gpus = local["gpu_count"].as_u64().unwrap_or(0);
        lines.push(format!("local: {host} — {cpus} CPUs, {ram:.0} GB RAM, {local_gpus} GPUs"));

        if let Some(gpu_arr) = local["gpus"].as_array() {
            for g in gpu_arr {
                let name = g["name"].as_str().unwrap_or("");
                let vram = g["vram_mb"].as_u64().unwrap_or(0);
                lines.push(format!("  GPU: {name} ({} GB VRAM)", vram / 1024));
            }
        }

        if !slurm.is_null() {
            let gpu_parts = summary["slurm_gpu_partitions"].as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>().join(", "))
                .unwrap_or_default();
            lines.push(format!("slurm: available — GPU partitions: [{gpu_parts}]"));

            if let Some(parts) = slurm["partitions"].as_array() {
                for p in parts {
                    let name = p["name"].as_str().unwrap_or("");
                    let cpus = p["cpu_cores"].as_u64().unwrap_or(0);
                    let gpus = p["gpus"].as_array().map(|a| a.len()).unwrap_or(0);
                    if cpus > 0 || gpus > 0 {
                        lines.push(format!("  partition {name}: {cpus} CPUs, {gpus} GPU type(s)"));
                    }
                }
            }
        } else {
            lines.push("slurm: not available".into());
        }

        // Routing recommendations for Adhvaryu
        lines.push(String::new());
        lines.push("routing recommendations:".into());
        if local_gpus > 0 {
            lines.push("  GPU workloads: run locally (GPU available)".into());
        }
        if !slurm.is_null() {
            let gpu_parts = summary["slurm_gpu_partitions"].as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
                .unwrap_or_default();
            if !gpu_parts.is_empty() {
                lines.push(format!("  Large GPU workloads: sbatch --partition={} --gres=gpu:1", gpu_parts[0]));
            }
            lines.push("  CPU-only HPC: sbatch without --gres".into());
        }
        lines.push(format!("  Quick analysis: run locally ({cpus} cores available)"));

        let content = lines.join("\n");

        // Store in chitta
        let mut chitta = ctx.chitta.lock().await;
        if chitta.connect().await.is_ok() {
            let _ = chitta.remember(
                &content,
                "wisdom",
                &["resource-topology", "compute", "chitta-research"],
                0.95,
            ).await;
            tracing::info!("scout: resource topology stored in chitta");
        }

        // Also log the summary
        tracing::info!(
            local_cpus = cpus,
            local_gpus,
            has_slurm = !slurm.is_null(),
            "scout: resource discovery complete"
        );

        Ok(AgentAction::Noop)
    }
}
