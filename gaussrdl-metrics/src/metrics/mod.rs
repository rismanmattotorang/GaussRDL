use std::fmt::Debug;
use gaussrdl_core::Error;

/// Trait for evaluation metrics
pub trait Metric: Debug + Send + Sync {
    /// Get the metric name
    fn name(&self) -> &str;
    
    /// Get the metric description
    fn description(&self) -> &str;
    
    /// Calculate the metric value
    fn calculate(&self, predictions: &[f64], targets: &[f64]) -> Result<f64, Error>;
}

/// Mean Absolute Error metric
#[derive(Debug)]
pub struct MAE;

impl MAE {
    pub fn new() -> Self {
        Self
    }
}

impl Metric for MAE {
    fn name(&self) -> &str {
        "mae"
    }
    
    fn description(&self) -> &str {
        "Mean Absolute Error"
    }
    
    fn calculate(&self, predictions: &[f64], targets: &[f64]) -> Result<f64, Error> {
        if predictions.len() != targets.len() {
            return Err(Error::validation(
                "Predictions and targets must have same length"
            ));
        }
        
        let sum: f64 = predictions.iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).abs())
            .sum();
            
        Ok(sum / predictions.len() as f64)
    }
}

/// Root Mean Square Error metric
#[derive(Debug)]
pub struct RMSE;

impl RMSE {
    pub fn new() -> Self {
        Self
    }
}

impl Metric for RMSE {
    fn name(&self) -> &str {
        "rmse"
    }
    
    fn description(&self) -> &str {
        "Root Mean Square Error"
    }
    
    fn calculate(&self, predictions: &[f64], targets: &[f64]) -> Result<f64, Error> {
        if predictions.len() != targets.len() {
            return Err(Error::validation(
                "Predictions and targets must have same length"
            ));
        }
        
        let sum: f64 = predictions.iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum();
            
        Ok((sum / predictions.len() as f64).sqrt())
    }
}

/// Mean Average Precision metric
#[derive(Debug)]
pub struct MAP {
    k: usize,
}

impl MAP {
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl Metric for MAP {
    fn name(&self) -> &str {
        "map"
    }
    
    fn description(&self) -> &str {
        "Mean Average Precision"
    }
    
    fn calculate(&self, predictions: &[f64], targets: &[f64]) -> Result<f64, Error> {
        if predictions.len() != targets.len() {
            return Err(Error::validation(
                "Predictions and targets must have same length"
            ));
        }
        
        // Create index vector and sort by predictions
        let mut indices: Vec<usize> = (0..predictions.len()).collect();
        indices.sort_by(|&i, &j| predictions[j].partial_cmp(&predictions[i]).unwrap());
        
        // Calculate precision at k
        let mut sum = 0.0;
        let mut num_relevant = 0;
        
        for (i, &idx) in indices.iter().take(self.k).enumerate() {
            if targets[idx] > 0.0 {
                num_relevant += 1;
                sum += num_relevant as f64 / (i + 1) as f64;
            }
        }
        
        Ok(if num_relevant > 0 {
            sum / num_relevant as f64
        } else {
            0.0
        })
    }
}

/// Area Under ROC Curve metric
#[derive(Debug)]
pub struct AUROC;

impl AUROC {
    pub fn new() -> Self {
        Self
    }
}

impl Metric for AUROC {
    fn name(&self) -> &str {
        "auroc"
    }
    
    fn description(&self) -> &str {
        "Area Under ROC Curve"
    }
    
    fn calculate(&self, predictions: &[f64], targets: &[f64]) -> Result<f64, Error> {
        if predictions.len() != targets.len() {
            return Err(Error::validation(
                "Predictions and targets must have same length"
            ));
        }
        
        // Create index vector and sort by predictions
        let mut indices: Vec<usize> = (0..predictions.len()).collect();
        indices.sort_by(|&i, &j| predictions[j].partial_cmp(&predictions[i]).unwrap());
        
        let mut tp = 0;
        let mut fp = 0;
        let total_p: usize = targets.iter().map(|&t| if t > 0.0 { 1 } else { 0 }).sum();
        let total_n = targets.len() - total_p;
        
        let mut auc = 0.0;
        let mut prev_tpr = 0.0;
        let mut prev_fpr = 0.0;
        
        for &idx in indices.iter() {
            if targets[idx] > 0.0 {
                tp += 1;
            } else {
                fp += 1;
            }
            
            let tpr = tp as f64 / total_p as f64;
            let fpr = fp as f64 / total_n as f64;
            
            auc += (tpr + prev_tpr) * (fpr - prev_fpr) / 2.0;
            
            prev_tpr = tpr;
            prev_fpr = fpr;
        }
        
        Ok(auc)
    }
}

/// Accuracy metric
#[derive(Debug)]
pub struct Accuracy;

impl Accuracy {
    pub fn new() -> Self {
        Self
    }
}

impl Metric for Accuracy {
    fn name(&self) -> &str {
        "accuracy"
    }
    
    fn description(&self) -> &str {
        "Accuracy"
    }
    
    fn calculate(&self, predictions: &[f64], targets: &[f64]) -> Result<f64, Error> {
        if predictions.len() != targets.len() {
            return Err(Error::validation(
                "Predictions and targets must have same length"
            ));
        }
        
        let correct = predictions.iter()
            .zip(targets.iter())
            .filter(|&(p, t)| (p.round() - t).abs() < f64::EPSILON)
            .count();
            
        Ok(correct as f64 / predictions.len() as f64)
    }
}

/// F1 Score metric
#[derive(Debug)]
pub struct F1Score;

impl F1Score {
    pub fn new() -> Self {
        Self
    }
}

impl Metric for F1Score {
    fn name(&self) -> &str {
        "f1"
    }
    
    fn description(&self) -> &str {
        "F1 Score"
    }
    
    fn calculate(&self, predictions: &[f64], targets: &[f64]) -> Result<f64, Error> {
        if predictions.len() != targets.len() {
            return Err(Error::validation(
                "Predictions and targets must have same length"
            ));
        }
        
        let tp = predictions.iter().zip(targets.iter()).filter(|&(&p, &t)| p.round() == 1.0 && t == 1.0).count() as f64;
        let fp = predictions.iter().zip(targets.iter()).filter(|&(&p, &t)| p.round() == 1.0 && t == 0.0).count() as f64;
        let fn_ = predictions.iter().zip(targets.iter()).filter(|&(&p, &t)| p.round() == 0.0 && t == 1.0).count() as f64;
        
        let precision = if (tp + fp) > 0.0 { tp / (tp + fp) } else { 0.0 };
        let recall = if (tp + fn_) > 0.0 { tp / (tp + fn_) } else { 0.0 };
        
        Ok(if (precision + recall) > 0.0 { 2.0 * (precision * recall) / (precision + recall) } else { 0.0 })
    }
}

/// Mean Squared Error metric
#[derive(Debug)]
pub struct MSE;

impl MSE {
    pub fn new() -> Self {
        Self
    }
}

impl Metric for MSE {
    fn name(&self) -> &str {
        "mse"
    }
    
    fn description(&self) -> &str {
        "Mean Squared Error"
    }
    
    fn calculate(&self, predictions: &[f64], targets: &[f64]) -> Result<f64, Error> {
        if predictions.len() != targets.len() {
            return Err(Error::validation(
                "Predictions and targets must have same length"
            ));
        }
        
        let sum: f64 = predictions.iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum();
            
        Ok(sum / predictions.len() as f64)
    }
}

/// Mean Reciprocal Rank metric
#[derive(Debug)]
pub struct MRR;

impl MRR {
    pub fn new() -> Self {
        Self
    }
}

impl Metric for MRR {
    fn name(&self) -> &str {
        "mrr"
    }
    
    fn description(&self) -> &str {
        "Mean Reciprocal Rank"
    }
    
    fn calculate(&self, predictions: &[f64], targets: &[f64]) -> Result<f64, Error> {
        if predictions.len() != targets.len() {
            return Err(Error::validation(
                "Predictions and targets must have same length"
            ));
        }
        
        let mut indices: Vec<usize> = (0..predictions.len()).collect();
        indices.sort_by(|&i, &j| predictions[j].partial_cmp(&predictions[i]).unwrap());
        
        for (i, &idx) in indices.iter().enumerate() {
            if targets[idx] > 0.0 {
                return Ok(1.0 / (i + 1) as f64);
            }
        }
        
        Ok(0.0)
    }
}

/// Normalized Discounted Cumulative Gain metric
#[derive(Debug)]
pub struct NDCG {
    k: usize,
}

impl NDCG {
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl Metric for NDCG {
    fn name(&self) -> &str {
        "ndcg"
    }
    
    fn description(&self) -> &str {
        "Normalized Discounted Cumulative Gain"
    }
    
    fn calculate(&self, predictions: &[f64], targets: &[f64]) -> Result<f64, Error> {
        if predictions.len() != targets.len() {
            return Err(Error::validation(
                "Predictions and targets must have same length"
            ));
        }
        
        let mut indices: Vec<usize> = (0..predictions.len()).collect();
        indices.sort_by(|&i, &j| predictions[j].partial_cmp(&predictions[i]).unwrap());
        
        let mut dcg = 0.0;
        for (i, &idx) in indices.iter().take(self.k).enumerate() {
            dcg += targets[idx] / ((i + 2) as f64).log2();
        }
        
        let mut ideal_indices: Vec<usize> = (0..targets.len()).collect();
        ideal_indices.sort_by(|&i, &j| targets[j].partial_cmp(&targets[i]).unwrap());
        
        let mut idcg = 0.0;
        for (i, &idx) in ideal_indices.iter().take(self.k).enumerate() {
            idcg += targets[idx] / ((i + 2) as f64).log2();
        }
        
        Ok(if idcg > 0.0 { dcg / idcg } else { 0.0 })
    }
}

/// Area Under Precision-Recall Curve metric
#[derive(Debug)]
pub struct AUPRC;

impl AUPRC {
    pub fn new() -> Self {
        Self
    }
}

impl Metric for AUPRC {
    fn name(&self) -> &str {
        "auprc"
    }
    
    fn description(&self) -> &str {
        "Area Under Precision-Recall Curve"
    }
    
    fn calculate(&self, predictions: &[f64], targets: &[f64]) -> Result<f64, Error> {
        if predictions.len() != targets.len() {
            return Err(Error::validation(
                "Predictions and targets must have same length"
            ));
        }
        
        // Create index vector and sort by predictions (descending)
        let mut indices: Vec<usize> = (0..predictions.len()).collect();
        indices.sort_by(|&i, &j| predictions[j].partial_cmp(&predictions[i]).unwrap());
        
        let mut tp = 0;
        let mut fp = 0;
        let total_p: usize = targets.iter().map(|&t| if t > 0.0 { 1 } else { 0 }).sum();
        
        let mut auc = 0.0;
        let mut prev_recall = 0.0;
        
        for &idx in indices.iter() {
            if targets[idx] > 0.0 {
                tp += 1;
            } else {
                fp += 1;
            }
            
            let precision = tp as f64 / (tp + fp) as f64;
            let recall = tp as f64 / total_p as f64;
            
            // Trapezoidal rule
            auc += precision * (recall - prev_recall);
            prev_recall = recall;
        }
        
        Ok(auc)
    }
}

/// R-squared (coefficient of determination) metric
#[derive(Debug)]
pub struct R2;

impl R2 {
    pub fn new() -> Self {
        Self
    }
}

impl Metric for R2 {
    fn name(&self) -> &str {
        "r2"
    }
    
    fn description(&self) -> &str {
        "R-squared (Coefficient of Determination)"
    }
    
    fn calculate(&self, predictions: &[f64], targets: &[f64]) -> Result<f64, Error> {
        if predictions.len() != targets.len() {
            return Err(Error::validation(
                "Predictions and targets must have same length"
            ));
        }
        
        let mean_target: f64 = targets.iter().sum::<f64>() / targets.len() as f64;
        
        let ss_res: f64 = predictions.iter()
            .zip(targets.iter())
            .map(|(p, t)| (t - p).powi(2))
            .sum();
            
        let ss_tot: f64 = targets.iter()
            .map(|t| (t - mean_target).powi(2))
            .sum();
            
        Ok(1.0 - (ss_res / ss_tot))
    }
}

/// Macro F1 Score for multiclass classification
#[derive(Debug)]
pub struct MacroF1 {
    num_classes: usize,
}

impl MacroF1 {
    pub fn new(num_classes: usize) -> Self {
        Self { num_classes }
    }
}

impl Metric for MacroF1 {
    fn name(&self) -> &str {
        "macro_f1"
    }
    
    fn description(&self) -> &str {
        "Macro-averaged F1 Score"
    }
    
    fn calculate(&self, predictions: &[f64], targets: &[f64]) -> Result<f64, Error> {
        if predictions.len() != targets.len() {
            return Err(Error::validation(
                "Predictions and targets must have same length"
            ));
        }
        
        let mut f1_scores = Vec::new();
        
        for class in 0..self.num_classes {
            let mut tp = 0;
            let mut fp = 0;
            let mut fn_count = 0;
            
            for i in 0..predictions.len() {
                let pred_class = predictions[i].round() as usize;
                let true_class = targets[i].round() as usize;
                
                if true_class == class && pred_class == class {
                    tp += 1;
                } else if pred_class == class {
                    fp += 1;
                } else if true_class == class {
                    fn_count += 1;
                }
            }
            
            let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
            let recall = if tp + fn_count > 0 { tp as f64 / (tp + fn_count) as f64 } else { 0.0 };
            let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
            
            f1_scores.push(f1);
        }
        
        Ok(f1_scores.iter().sum::<f64>() / f1_scores.len() as f64)
    }
}

/// Link Prediction Recall
#[derive(Debug)]
pub struct LinkPredictionRecall {
    k: usize,
}

impl LinkPredictionRecall {
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl Metric for LinkPredictionRecall {
    fn name(&self) -> &str {
        "link_prediction_recall"
    }
    
    fn description(&self) -> &str {
        "Link Prediction Recall@K"
    }
    
    fn calculate(&self, predictions: &[f64], targets: &[f64]) -> Result<f64, Error> {
        if predictions.len() != targets.len() {
            return Err(Error::validation(
                "Predictions and targets must have same length"
            ));
        }
        
        // Create index vector and sort by predictions (descending)
        let mut indices: Vec<usize> = (0..predictions.len()).collect();
        indices.sort_by(|&i, &j| predictions[j].partial_cmp(&predictions[i]).unwrap());
        
        let total_relevant: usize = targets.iter().map(|&t| if t > 0.0 { 1 } else { 0 }).sum();
        
        if total_relevant == 0 {
            return Ok(0.0);
        }
        
        let mut relevant_retrieved = 0;
        for &idx in indices.iter().take(self.k) {
            if targets[idx] > 0.0 {
                relevant_retrieved += 1;
            }
        }
        
        Ok(relevant_retrieved as f64 / total_relevant as f64)
    }
}

/// Link Prediction Precision
#[derive(Debug)]  
pub struct LinkPredictionPrecision {
    k: usize,
}

impl LinkPredictionPrecision {
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl Metric for LinkPredictionPrecision {
    fn name(&self) -> &str {
        "link_prediction_precision"
    }
    
    fn description(&self) -> &str {
        "Link Prediction Precision@K"
    }
    
    fn calculate(&self, predictions: &[f64], targets: &[f64]) -> Result<f64, Error> {
        if predictions.len() != targets.len() {
            return Err(Error::validation(
                "Predictions and targets must have same length"
            ));
        }
        
        // Create index vector and sort by predictions (descending)
        let mut indices: Vec<usize> = (0..predictions.len()).collect();
        indices.sort_by(|&i, &j| predictions[j].partial_cmp(&predictions[i]).unwrap());
        
        let mut relevant_retrieved = 0;
        for &idx in indices.iter().take(self.k) {
            if targets[idx] > 0.0 {
                relevant_retrieved += 1;
            }
        }
        
        Ok(relevant_retrieved as f64 / self.k.min(predictions.len()) as f64)
    }
}

/// Average Precision Score
#[derive(Debug)]
pub struct AveragePrecision;

impl AveragePrecision {
    pub fn new() -> Self {
        Self
    }
}

impl Metric for AveragePrecision {
    fn name(&self) -> &str {
        "average_precision"
    }
    
    fn description(&self) -> &str {
        "Average Precision Score"
    }
    
    fn calculate(&self, predictions: &[f64], targets: &[f64]) -> Result<f64, Error> {
        if predictions.len() != targets.len() {
            return Err(Error::validation(
                "Predictions and targets must have same length"
            ));
        }
        
        // Create index vector and sort by predictions (descending)
        let mut indices: Vec<usize> = (0..predictions.len()).collect();
        indices.sort_by(|&i, &j| predictions[j].partial_cmp(&predictions[i]).unwrap());
        
        let mut tp = 0;
        let mut fp = 0;
        let mut ap = 0.0;
        let total_p: usize = targets.iter().map(|&t| if t > 0.0 { 1 } else { 0 }).sum();
        
        if total_p == 0 {
            return Ok(0.0);
        }
        
        for &idx in indices.iter() {
            if targets[idx] > 0.0 {
                tp += 1;
                let precision = tp as f64 / (tp + fp) as f64;
                ap += precision;
            } else {
                fp += 1;
            }
        }
        
        Ok(ap / total_p as f64)
    }
} 