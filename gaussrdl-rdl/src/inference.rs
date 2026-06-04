//! Inference key performance / accuracy indicators.
//!
//! Reports both *performance* KPIs (latency, throughput) and *accuracy* KPIs
//! appropriate to the task: ROC-AUC / accuracy / Brier score for
//! classification, MAE / RMSE / R² for regression.

use crate::metrics::{accuracy_from_logits, mae, rmse, roc_auc};
use crate::task::TaskType;
use serde::{Deserialize, Serialize};

/// Inference report over a held-out set.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InferenceReport {
    pub num_samples: usize,
    /// Wall-clock time for the forward pass (whole-graph), in milliseconds.
    pub latency_ms: f32,
    /// Predictions per second.
    pub throughput_per_s: f32,

    // Classification KPIs.
    pub auroc: f32,
    pub accuracy: f32,
    /// Brier score (mean squared error of probabilities); lower is better.
    pub brier: f32,
    pub positive_rate: f32,

    // Regression KPIs.
    pub mae: f32,
    pub rmse: f32,
    /// Coefficient of determination; higher is better.
    pub r2: f32,
}

impl InferenceReport {
    /// Build a report. `scores` are sigmoid probabilities (classification) or
    /// raw predictions (regression); `logits` are raw logits (classification).
    pub fn build(
        task: TaskType,
        scores: &[f32],
        logits: &[f32],
        labels: &[f32],
        latency_ms: f32,
    ) -> Self {
        let n = labels.len();
        let mut r = InferenceReport {
            num_samples: n,
            latency_ms,
            throughput_per_s: if latency_ms > 0.0 { n as f32 / (latency_ms / 1000.0) } else { 0.0 },
            ..Default::default()
        };
        match task {
            TaskType::BinaryClassification => {
                r.auroc = roc_auc(scores, labels);
                r.accuracy = accuracy_from_logits(logits, labels);
                r.brier = scores
                    .iter()
                    .zip(labels)
                    .map(|(p, y)| (p - y) * (p - y))
                    .sum::<f32>()
                    / n.max(1) as f32;
                r.positive_rate = labels.iter().filter(|&&y| y > 0.5).count() as f32 / n.max(1) as f32;
            }
            TaskType::Regression => {
                r.mae = mae(scores, labels);
                r.rmse = rmse(scores, labels);
                let mean = labels.iter().sum::<f32>() / n.max(1) as f32;
                let ss_tot: f32 = labels.iter().map(|y| (y - mean) * (y - mean)).sum();
                let ss_res: f32 = scores.iter().zip(labels).map(|(p, y)| (p - y) * (p - y)).sum();
                r.r2 = if ss_tot > 1e-9 { 1.0 - ss_res / ss_tot } else { 0.0 };
            }
        }
        r
    }
}
