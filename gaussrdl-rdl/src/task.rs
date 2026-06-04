//! Task definitions, prediction heads, and losses.

use crate::error::Result;
use candle_core::Tensor;
use candle_nn::loss::mse;
use candle_nn::{linear, Linear, Module, VarBuilder};

/// Supported predictive task types (RelBench entity-level).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TaskType {
    /// Binary node classification (metric: ROC-AUC).
    BinaryClassification,
    /// Node regression (metric: MAE / RMSE).
    Regression,
}

impl TaskType {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().replace(['-', '_'], "").as_str() {
            "churn" | "classification" | "binary" | "clf" => Some(TaskType::BinaryClassification),
            "ltv" | "regression" | "reg" => Some(TaskType::Regression),
            _ => None,
        }
    }
}

/// A linear prediction head mapping node embeddings to a scalar output.
pub struct TaskHead {
    lin: Linear,
}

impl TaskHead {
    pub fn new(in_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self { lin: linear(in_dim, 1, vb.pp("head"))? })
    }

    /// [n, in_dim] -> [n] scalar predictions (logits for classification).
    pub fn forward(&self, z: &Tensor) -> Result<Tensor> {
        Ok(self.lin.forward(z)?.squeeze(1)?)
    }
}

/// Compute the training loss for the given task.
pub fn loss(task: TaskType, preds: &Tensor, targets: &Tensor) -> Result<Tensor> {
    Ok(match task {
        TaskType::BinaryClassification => bce_with_logits(preds, targets)?,
        TaskType::Regression => mse(preds, targets)?,
    })
}

/// Numerically stable binary cross-entropy from logits.
///
/// `loss = relu(x) - x*z + log(1 + exp(-|x|))`, which never overflows for
/// large-magnitude logits (unlike `sigmoid(x).log()`).
pub fn bce_with_logits(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let relu = logits.relu()?;
    let xz = (logits * targets)?;
    // |x| = relu(x) + relu(-x)
    let abs = (logits.relu()? + logits.neg()?.relu()?)?;
    let softplus = ((abs.neg()?.exp()? + 1.0)?).log()?; // log(1 + exp(-|x|))
    let per_elem = ((relu - xz)? + softplus)?;
    Ok(per_elem.mean_all()?)
}
