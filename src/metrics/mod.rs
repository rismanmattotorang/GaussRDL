use std::fmt;
use ndarray::{Array1, ArrayView1, Array2};
use crate::error::Result;
use std::f64;

/// Metric trait for evaluation metrics
pub trait Metric: fmt::Debug + Send + Sync {
    /// Gets the metric name
    fn name(&self) -> &str;
    
    /// Gets the metric description
    fn description(&self) -> &str;
    
    /// Computes the metric value
    fn compute(&self, true_values: &Array1<f64>, pred_values: &Array1<f64>) -> Result<f64>;
    
    /// Whether higher values are better
    fn higher_is_better(&self) -> bool;
}

/// Mean Absolute Error metric
#[derive(Debug, Clone)]
pub struct MAE;

impl Metric for MAE {
    fn name(&self) -> &str {
        "mae"
    }
    
    fn description(&self) -> &str {
        "Mean Absolute Error"
    }
    
    fn compute(&self, true_values: &Array1<f64>, pred_values: &Array1<f64>) -> Result<f64> {
        Ok((true_values - pred_values).mapv(f64::abs).mean().unwrap_or(0.0))
    }
    
    fn higher_is_better(&self) -> bool {
        false
    }
}

/// Mean Squared Error metric
#[derive(Debug, Clone)]
pub struct MSE;

impl Metric for MSE {
    fn name(&self) -> &str {
        "mse"
    }
    
    fn description(&self) -> &str {
        "Mean Squared Error"
    }
    
    fn compute(&self, true_values: &Array1<f64>, pred_values: &Array1<f64>) -> Result<f64> {
        Ok((true_values - pred_values).mapv(|x| x * x).mean().unwrap_or(0.0))
    }
    
    fn higher_is_better(&self) -> bool {
        false
    }
}

/// Root Mean Squared Error metric
#[derive(Debug, Clone)]
pub struct RMSE;

impl Metric for RMSE {
    fn name(&self) -> &str {
        "rmse"
    }
    
    fn description(&self) -> &str {
        "Root Mean Squared Error"
    }
    
    fn compute(&self, true_values: &Array1<f64>, pred_values: &Array1<f64>) -> Result<f64> {
        let mse = MSE.compute(true_values, pred_values)?;
        Ok(mse.sqrt())
    }
    
    fn higher_is_better(&self) -> bool {
        false
    }
}

/// Area Under ROC Curve
#[derive(Debug, Clone)]
pub struct AUROC;

impl AUROC {
    fn compute_roc(&self, true_values: &Array1<f64>, pred_values: &Array1<f64>) -> (Vec<f64>, Vec<f64>) {
        let mut pairs: Vec<_> = true_values.iter().zip(pred_values.iter()).collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        
        let n_pos = true_values.iter().filter(|&&x| x > 0.5).count() as f64;
        let n_neg = true_values.len() as f64 - n_pos;
        
        let mut tpr = vec![0.0];
        let mut fpr = vec![0.0];
        let mut tp = 0.0;
        let mut fp = 0.0;
        
        for (true_val, _) in pairs {
            if *true_val > 0.5 {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            tpr.push(tp / n_pos);
            fpr.push(fp / n_neg);
        }
        
        (tpr, fpr)
    }
}

impl Metric for AUROC {
    fn name(&self) -> &str {
        "auroc"
    }
    
    fn description(&self) -> &str {
        "Area Under ROC Curve"
    }
    
    fn compute(&self, true_values: &Array1<f64>, pred_values: &Array1<f64>) -> Result<f64> {
        let (tpr, fpr) = self.compute_roc(true_values, pred_values);
        let mut auc = 0.0;
        
        for i in 1..tpr.len() {
            auc += (tpr[i] + tpr[i-1]) * (fpr[i] - fpr[i-1]) / 2.0;
        }
        
        Ok(auc)
    }
    
    fn higher_is_better(&self) -> bool {
        true
    }
}

/// Average Precision metric
#[derive(Debug, Clone)]
pub struct AveragePrecision;

impl Metric for AveragePrecision {
    fn name(&self) -> &str {
        "ap"
    }
    
    fn description(&self) -> &str {
        "Average Precision"
    }
    
    fn compute(&self, true_values: &Array1<f64>, pred_values: &Array1<f64>) -> Result<f64> {
        let mut pairs: Vec<_> = true_values.iter().zip(pred_values.iter()).collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        
        let n_pos = true_values.iter().filter(|&&x| x > 0.5).count() as f64;
        let mut tp = 0.0;
        let mut sum_prec = 0.0;
        
        for (i, (true_val, _)) in pairs.iter().enumerate() {
            if **true_val > 0.5 {
                tp += 1.0;
                sum_prec += tp / (i as f64 + 1.0);
            }
        }
        
        Ok(if n_pos > 0.0 { sum_prec / n_pos } else { 0.0 })
    }
    
    fn higher_is_better(&self) -> bool {
        true
    }
}

/// Accuracy metric
#[derive(Debug, Clone)]
pub struct Accuracy {
    pub threshold: f64,
}

impl Metric for Accuracy {
    fn name(&self) -> &str {
        "accuracy"
    }
    
    fn description(&self) -> &str {
        "Classification Accuracy"
    }
    
    fn compute(&self, true_values: &Array1<f64>, pred_values: &Array1<f64>) -> Result<f64> {
        let pred_binary = pred_values.mapv(|x| if x >= self.threshold { 1.0 } else { 0.0 });
        let true_binary = true_values.mapv(|x| if x >= self.threshold { 1.0 } else { 0.0 });
        
        Ok((pred_binary.iter().zip(true_binary.iter())
            .filter(|(&p, &t)| p == t)
            .count() as f64) / true_values.len() as f64)
    }
    
    fn higher_is_better(&self) -> bool {
        true
    }
}

/// F1 Score metric
#[derive(Debug, Clone)]
pub struct F1Score {
    pub threshold: f64,
}

impl Metric for F1Score {
    fn name(&self) -> &str {
        "f1"
    }
    
    fn description(&self) -> &str {
        "F1 Score"
    }
    
    fn compute(&self, true_values: &Array1<f64>, pred_values: &Array1<f64>) -> Result<f64> {
        let pred_binary = pred_values.mapv(|x| if x >= self.threshold { 1.0 } else { 0.0 });
        let true_binary = true_values.mapv(|x| if x >= self.threshold { 1.0 } else { 0.0 });
        
        let tp = (&pred_binary * &true_binary).sum();
        let fp = (&pred_binary * &(1.0 - &true_binary)).sum();
        let fn_val = (&(1.0 - &pred_binary) * &true_binary).sum();
        
        let precision = tp / (tp + fp);
        let recall = tp / (tp + fn_val);
        
        Ok(if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        })
    }
    
    fn higher_is_better(&self) -> bool {
        true
    }
}

/// NDCG (Normalized Discounted Cumulative Gain)
#[derive(Debug, Clone)]
pub struct NDCG {
    pub k: usize,
}

impl Metric for NDCG {
    fn name(&self) -> &str {
        "ndcg"
    }
    
    fn description(&self) -> &str {
        "Normalized Discounted Cumulative Gain"
    }
    
    fn compute(&self, true_values: &Array1<f64>, pred_values: &Array1<f64>) -> Result<f64> {
        let mut pairs: Vec<_> = true_values.iter().zip(pred_values.iter()).collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        
        let dcg = pairs.iter()
            .take(self.k)
            .enumerate()
            .map(|(i, (t, _))| *t / (i as f64 + 2.0).log2())
            .sum::<f64>();
            
        let mut ideal_pairs: Vec<_> = true_values.iter().collect();
        ideal_pairs.sort_by(|a, b| b.partial_cmp(a).unwrap());
        
        let idcg = ideal_pairs.iter()
            .take(self.k)
            .enumerate()
            .map(|(i, t)| **t / (i as f64 + 2.0).log2())
            .sum::<f64>();
            
        Ok(if idcg > 0.0 { dcg / idcg } else { 0.0 })
    }
    
    fn higher_is_better(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_mae() {
        let true_values = array![1.0, 2.0, 3.0];
        let pred_values = array![0.8, 2.1, 2.9];
        let mae = MAE.compute(&true_values, &pred_values).unwrap();
        assert!((mae - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_auroc() {
        let true_values = array![1.0, 0.0, 1.0, 0.0];
        let pred_values = array![0.9, 0.1, 0.8, 0.2];
        let auroc = AUROC.compute(&true_values, &pred_values).unwrap();
        assert!(auroc > 0.9);
    }

    #[test]
    fn test_accuracy() {
        let true_values = array![1.0, 0.0, 1.0, 0.0];
        let pred_values = array![0.9, 0.1, 0.8, 0.2];
        let acc = Accuracy { threshold: 0.5 }.compute(&true_values, &pred_values).unwrap();
        assert_eq!(acc, 1.0);
    }

    #[test]
    fn test_ndcg() {
        let true_values = array![3.0, 2.0, 1.0, 0.0];
        let pred_values = array![0.9, 0.8, 0.1, 0.2];
        let ndcg = NDCG { k: 10 }.compute(&true_values, &pred_values).unwrap();
        assert!(ndcg > 0.8);
    }
} 