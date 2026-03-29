use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use rusqlite::Connection;

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

pub struct SqliteTracker {
    conn: std::sync::Mutex<Connection>,
    run_id: String,
}

impl SqliteTracker {
    pub fn open(db_path: &str, run_id: &str) -> Result<Self, rusqlite::Error> {
        let conn = Connection::open(db_path)?;
        conn.execute_batch("
            CREATE TABLE IF NOT EXISTS token_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                agent TEXT NOT NULL,
                tokens_in INTEGER NOT NULL DEFAULT 0,
                tokens_out INTEGER NOT NULL DEFAULT 0,
                cost_usd REAL NOT NULL DEFAULT 0.0,
                ts TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_run ON token_usage(run_id);
        ")?;
        Ok(Self { conn: std::sync::Mutex::new(conn), run_id: run_id.to_string() })
    }

    pub fn record(&self, agent: &str, tokens_in: u64, tokens_out: u64, cost_usd: f64) {
        if let Ok(conn) = self.conn.lock() {
            let _ = conn.execute(
                "INSERT INTO token_usage (run_id, agent, tokens_in, tokens_out, cost_usd) VALUES (?1,?2,?3,?4,?5)",
                rusqlite::params![self.run_id, agent, tokens_in as i64, tokens_out as i64, cost_usd],
            );
        }
    }

    pub fn total_cost(&self) -> f64 {
        if let Ok(conn) = self.conn.lock() {
            conn.query_row(
                "SELECT COALESCE(SUM(cost_usd),0) FROM token_usage WHERE run_id=?1",
                rusqlite::params![self.run_id],
                |r| r.get(0),
            ).unwrap_or(0.0)
        } else { 0.0 }
    }
}

pub struct ResourceManager {
    gpu_slots: Arc<Semaphore>,
    cpu_slots: Arc<Semaphore>,
    budget_microdollars: Arc<AtomicU64>,
    tracker: Option<SqliteTracker>,
}

impl ResourceManager {
    pub fn new(gpu_count: usize, cpu_workers: usize, budget_usd: f64) -> Self {
        Self {
            gpu_slots: Arc::new(Semaphore::new(gpu_count)),
            cpu_slots: Arc::new(Semaphore::new(cpu_workers)),
            budget_microdollars: Arc::new(AtomicU64::new((budget_usd * 1_000_000.0) as u64)),
            tracker: None,
        }
    }

    pub fn with_tracking(gpu_count: usize, cpu_workers: usize, budget_usd: f64, db_path: &str, run_id: &str) -> Self {
        let tracker = SqliteTracker::open(db_path, run_id).ok();
        Self {
            gpu_slots: Arc::new(Semaphore::new(gpu_count)),
            cpu_slots: Arc::new(Semaphore::new(cpu_workers)),
            budget_microdollars: Arc::new(AtomicU64::new((budget_usd * 1_000_000.0) as u64)),
            tracker,
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

    pub fn charge_tracked(&self, cost_usd: f64, agent: &str, tokens_in: u64, tokens_out: u64) {
        self.charge(cost_usd);
        if let Some(tracker) = &self.tracker {
            tracker.record(agent, tokens_in, tokens_out, cost_usd);
        }
    }

    pub fn budget_exhausted(&self) -> bool {
        self.budget_microdollars.load(Ordering::Relaxed) == 0
    }

    pub fn remaining_budget(&self) -> f64 {
        self.budget_microdollars.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }
}
