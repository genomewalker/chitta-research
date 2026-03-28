use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Debug, thiserror::Error)]
pub enum ResourceError {
    #[error("budget exhausted")]
    BudgetExhausted,
    #[error("semaphore closed")]
    SemaphoreClosed,
}

pub struct ResourceSlot {
    _gpu: Option<OwnedSemaphorePermit>,
    _cpu: OwnedSemaphorePermit,
}

pub struct ResourceManager {
    gpu_slots: Arc<Semaphore>,
    cpu_slots: Arc<Semaphore>,
    budget_microdollars: Arc<AtomicU64>,
}

impl ResourceManager {
    pub fn new(gpu_count: usize, cpu_workers: usize, budget_usd: f64) -> Self {
        Self {
            gpu_slots: Arc::new(Semaphore::new(gpu_count)),
            cpu_slots: Arc::new(Semaphore::new(cpu_workers)),
            budget_microdollars: Arc::new(AtomicU64::new((budget_usd * 1_000_000.0) as u64)),
        }
    }

    pub async fn acquire(&self, needs_gpu: bool) -> Result<ResourceSlot, ResourceError> {
        if self.budget_exhausted() {
            return Err(ResourceError::BudgetExhausted);
        }

        let gpu_permit = if needs_gpu {
            Some(
                self.gpu_slots
                    .clone()
                    .acquire_owned()
                    .await
                    .map_err(|_| ResourceError::SemaphoreClosed)?,
            )
        } else {
            None
        };

        let cpu_permit = self
            .cpu_slots
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| ResourceError::SemaphoreClosed)?;

        Ok(ResourceSlot {
            _gpu: gpu_permit,
            _cpu: cpu_permit,
        })
    }

    pub fn charge(&self, cost_usd: f64) {
        let microdollars = (cost_usd * 1_000_000.0) as u64;
        self.budget_microdollars
            .fetch_sub(microdollars.min(self.budget_microdollars.load(Ordering::Relaxed)), Ordering::Relaxed);
    }

    pub fn budget_exhausted(&self) -> bool {
        self.budget_microdollars.load(Ordering::Relaxed) == 0
    }

    pub fn remaining_budget(&self) -> f64 {
        self.budget_microdollars.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }
}
