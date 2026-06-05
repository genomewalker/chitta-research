use std::collections::{BTreeMap, HashMap};

use cr_types::{
    Account, ActionId, ActionReceipt, Bid, ContextSnapshot, Genome, LineageId, LineageStatus,
    RewardEvent, SlotKind,
};
use serde::Serialize;
use uuid::Uuid;

/// Shadow-mode auction scheduler — Economy of Minds (arXiv:2606.02859).
///
/// In shadow mode (default): computes hypothetical winners + logs them.
/// Fixed dispatch continues unchanged. Promotes to live via shadow=false
/// once shadow log shows auction beats fixed dispatch on realized gain.
pub struct AuctionScheduler {
    pub accounts: HashMap<LineageId, Account>,
    /// Pending rewards sorted by ActionId for deterministic tick-boundary settlement.
    reward_buf: BTreeMap<ActionId, RewardEvent>,
    receipts: HashMap<ActionId, ActionReceipt>,
    /// shadow=true: log only, no cash mutation. shadow=false: live dispatch.
    pub shadow: bool,
    log: Vec<ShadowLogEntry>,
}

#[derive(Serialize)]
pub struct ShadowLogEntry {
    pub slot: SlotKind,
    pub snap: ContextSnapshot,
    pub winner: Option<LineageId>,
    pub clearing_price: f64,
    pub bid_count: usize,
}

impl AuctionScheduler {
    /// Seed with 3 Hotr lineages, 3 Adhvaryu, 2 Kriya — all with default genomes.
    pub fn new(shadow: bool) -> Self {
        let mut accounts = HashMap::new();
        for _ in 0..8 {
            let id = Uuid::now_v7();
            accounts.insert(id, Account::new(Genome::default(), id, 100.0));
        }
        Self { accounts, reward_buf: BTreeMap::new(), receipts: HashMap::new(), shadow, log: Vec::new() }
    }

    /// Second-price (Vickrey) auction for a slot.
    /// In shadow mode, logs the outcome without reserving cash.
    pub fn clear_slot(&mut self, slot: SlotKind, snap: &ContextSnapshot, bids: Vec<Bid>) -> Option<(LineageId, f64)> {
        let mut eligible: Vec<&Bid> = bids.iter()
            .filter(|b| b.slot == slot)
            .filter(|b| {
                self.accounts.get(&b.lineage_id)
                    .map(|a| matches!(a.status, LineageStatus::Live) && a.available() >= b.price)
                    .unwrap_or(false)
            })
            .collect();
        eligible.sort_by(|a, b| b.price.partial_cmp(&a.price).unwrap_or(std::cmp::Ordering::Equal));

        let winner = eligible.first()?;
        let clearing_price = eligible.get(1).map(|b| b.price).unwrap_or(0.0);

        self.log.push(ShadowLogEntry {
            slot,
            snap: snap.clone(),
            winner: Some(winner.lineage_id),
            clearing_price,
            bid_count: bids.len(),
        });

        if !self.shadow {
            if let Some(acc) = self.accounts.get_mut(&winner.lineage_id) {
                acc.reserved += clearing_price;
            }
        }

        Some((winner.lineage_id, clearing_price))
    }

    pub fn register_receipt(&mut self, receipt: ActionReceipt) {
        self.receipts.insert(receipt.action_id, receipt);
    }

    pub fn record_reward(&mut self, event: RewardEvent) {
        self.reward_buf.insert(event.action_id, event);
    }

    /// Drain reward_buf in ActionId order, credit via γ^hops eligibility trace.
    /// Keyed by lineage_id so credit lands on the genome, not the live slot.
    pub fn settle_tick(&mut self) {
        let events: Vec<RewardEvent> = self.reward_buf.values().cloned().collect();
        self.reward_buf.clear();
        for ev in events {
            let gamma = self.receipts.get(&ev.action_id)
                .and_then(|r| self.accounts.get(&r.lineage_id))
                .map(|a| a.genome.credit_decay_gamma)
                .unwrap_or(0.7);

            if let Some(receipt) = self.receipts.get(&ev.action_id).cloned() {
                self.credit_lineage(receipt.lineage_id, ev.empirical_gain);
                for (hop, parent_id) in receipt.parent_action_ids.iter().enumerate() {
                    if let Some(parent) = self.receipts.get(parent_id).cloned() {
                        let decay = gamma.powi(hop as i32 + 1);
                        self.credit_lineage(parent.lineage_id, ev.empirical_gain * decay);
                    }
                }
            }
        }
    }

    fn credit_lineage(&mut self, lineage_id: LineageId, amount: f64) {
        if let Some(acc) = self.accounts.get_mut(&lineage_id) {
            if !self.shadow {
                acc.cash += amount;
                acc.reserved = (acc.reserved - amount).max(0.0);
            }
        }
    }

    /// Bottom-third relative bankruptcy within the population.
    /// Only touches genomes (resamples to default), never substrate state.
    /// Skips lineages with free_runs_left > 0 (explorer subsidy).
    pub fn bankruptcy_pass(&mut self) {
        let mut ranked: Vec<(LineageId, f64)> = self.accounts.iter()
            .filter(|(_, a)| a.free_runs_left == 0 && matches!(a.status, LineageStatus::Live))
            .map(|(id, a)| (*id, a.cash))
            .collect();
        ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let bankrupt_n = (ranked.len() / 3).max(if ranked.is_empty() { 0 } else { 1 });
        for (id, _) in ranked.iter().take(bankrupt_n) {
            if let Some(acc) = self.accounts.get_mut(id) {
                acc.genome = Genome::default();
                acc.cash = 100.0;
                acc.free_runs_left = 3;
            }
        }
    }

    pub fn tick_free_runs(&mut self) {
        for acc in self.accounts.values_mut() {
            acc.free_runs_left = acc.free_runs_left.saturating_sub(1);
        }
    }

    pub fn shadow_log_len(&self) -> usize { self.log.len() }

    pub fn dump_shadow_log(&self, path: &str) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(&self.log)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}
