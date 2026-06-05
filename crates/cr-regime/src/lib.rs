use serde::{Deserialize, Serialize};

/// CUSUM changepoint detector on the edge-type usage distribution.
pub struct Cusum {
    pub k: f64,
    pub h: f64,
    hi: f64,
    lo: f64,
    pub last_changepoint: Option<u64>,
}

impl Cusum {
    pub fn new(k: f64, h: f64) -> Self {
        Self { k, h, hi: 0.0, lo: 0.0, last_changepoint: None }
    }

    /// Update with one observation. Returns true when a changepoint is detected.
    pub fn update(&mut self, observation_ll: f64, baseline_ll: f64, turn: u64) -> bool {
        let s = observation_ll - baseline_ll - self.k;
        self.hi = (self.hi + s).max(0.0);
        self.lo = (self.lo - s).max(0.0);
        if self.hi > self.h || self.lo > self.h {
            self.hi = 0.0;
            self.lo = 0.0;
            self.last_changepoint = Some(turn);
            true
        } else {
            false
        }
    }

    pub fn reset(&mut self) {
        self.hi = 0.0;
        self.lo = 0.0;
    }
}

/// Tracks total description length as a Lyapunov certificate.
/// A schema turn is non-converging iff adjusted DL (DL minus accrued switch budget) rises.
pub struct LyapunovTracker {
    pub switch_budget: f64,
    history: Vec<f64>,
}

impl LyapunovTracker {
    pub fn new() -> Self {
        Self { switch_budget: 0.0, history: vec![] }
    }

    pub fn record(&mut self, dl: f64, switch_cost: f64) {
        self.switch_budget += switch_cost;
        self.history.push(dl);
    }

    pub fn is_converging(&self, window: usize) -> bool {
        if self.history.len() < window + 1 { return true; }
        let recent = &self.history[self.history.len() - window..];
        let adjusted: Vec<f64> = recent.iter().enumerate()
            .map(|(i, &dl)| dl - self.switch_budget * (i as f64 / window as f64))
            .collect();
        adjusted.windows(2).all(|w| w[1] <= w[0] + 1e-6)
    }

    pub fn total_dl(&self) -> f64 {
        self.history.last().copied().unwrap_or(0.0)
    }
}

impl Default for LyapunovTracker {
    fn default() -> Self { Self::new() }
}

/// Combines CUSUM + Lyapunov into per-turn regime event emission.
pub struct RegimeDetector {
    pub cusum: Cusum,
    pub lyapunov: LyapunovTracker,
    turn: u64,
}

impl RegimeDetector {
    pub fn new() -> Self {
        Self { cusum: Cusum::new(0.5, 5.0), lyapunov: LyapunovTracker::new(), turn: 0 }
    }

    pub fn process(
        &mut self,
        dl: f64,
        switch_cost: f64,
        observation_ll: f64,
        baseline_ll: f64,
    ) -> Vec<RegimeEvent> {
        self.turn += 1;
        let mut events = vec![];
        self.lyapunov.record(dl, switch_cost);
        if !self.lyapunov.is_converging(5) {
            events.push(RegimeEvent::LyapunovViolation {
                turn: self.turn,
                excess: dl - self.lyapunov.switch_budget,
            });
        }
        if self.cusum.update(observation_ll, baseline_ll, self.turn) {
            events.push(RegimeEvent::UsageChangepoint {
                at_turn: self.turn,
                stat: self.cusum.hi.max(self.cusum.lo),
            });
        }
        events
    }
}

impl Default for RegimeDetector {
    fn default() -> Self { Self::new() }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegimeEvent {
    VocabExpanded { name: String },
    VocabContracted { name: String },
    UsageChangepoint { at_turn: u64, stat: f64 },
    LyapunovViolation { turn: u64, excess: f64 },
}
