// Metrics Evaluation Example
// This example demonstrates how to use different evaluation metrics similar to RelBench Python examples

use gaussrdl::GaussRDLResult;
use std::time::Instant;

#[tokio::main]
async fn main() -> GaussRDLResult<()> {
    println!("🚀 GaussRDL Metrics Evaluation Example");
    println!("=======================================");
    
    // Example 1: Classification Metrics
    println!("\n1. Classification Metrics");
    println!("=========================");
    evaluate_classification_metrics().await?;
    
    // Example 2: Regression Metrics
    println!("\n2. Regression Metrics");
    println!("====================");
    evaluate_regression_metrics().await?;
    
    // Example 3: Ranking Metrics
    println!("\n3. Ranking Metrics");
    println!("==================");
    evaluate_ranking_metrics().await?;
    
    // Example 4: Link Prediction Metrics (commented out due to API limitations)
    println!("\n4. Link Prediction Metrics");
    println!("==========================");
    evaluate_link_prediction_metrics().await?;
    
    // Example 5: Metrics Comparison
    println!("\n5. Metrics Performance Comparison");
    println!("=================================");
    compare_metrics_performance().await?;
    
    println!("\n🎉 Metrics evaluation examples completed!");
    Ok(())
}

async fn evaluate_classification_metrics() -> GaussRDLResult<()> {
    // Sample binary classification data
    let predictions = vec![0.9, 0.2, 0.8, 0.1, 0.7, 0.3, 0.95, 0.05];
    let targets = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    
    println!("Sample data: {} predictions vs {} targets", predictions.len(), targets.len());
    
    // AUROC - using available methods
    let auroc_score = calculate_auroc(&predictions, &targets)?;
    println!("  📊 AUROC: {:.4}", auroc_score);
    
    // Accuracy
    let accuracy_score = calculate_accuracy(&predictions, &targets)?;
    println!("  🎯 Accuracy: {:.4}", accuracy_score);
    
    // Precision (simulated)
    let precision_score = calculate_precision(&predictions, &targets)?;
    println!("  🔍 Precision: {:.4}", precision_score);
    
    // Recall (simulated)
    let recall_score = calculate_recall(&predictions, &targets)?;
    println!("  📈 Recall: {:.4}", recall_score);
    
    // F1 Score
    let f1_score = calculate_f1(&predictions, &targets)?;
    println!("  ⚖️  F1 Score: {:.4}", f1_score);
    
    // MacroF1 (simulated)
    let macro_f1_score = calculate_macro_f1(&predictions, &targets)?;
    println!("  📊 Macro F1: {:.4}", macro_f1_score);
    
    Ok(())
}

async fn evaluate_regression_metrics() -> GaussRDLResult<()> {
    // Sample regression data
    let predictions = vec![2.5, 3.2, 1.8, 4.1, 2.9, 3.7, 1.5, 4.3];
    let targets = vec![2.3, 3.1, 2.0, 4.0, 3.0, 3.5, 1.7, 4.1];
    
    println!("Sample data: {} predictions vs {} targets", predictions.len(), targets.len());
    
    // RMSE
    let rmse_score = calculate_rmse(&predictions, &targets)?;
    println!("  📐 RMSE: {:.4}", rmse_score);
    
    // MAE
    let mae_score = calculate_mae(&predictions, &targets)?;
    println!("  📏 MAE: {:.4}", mae_score);
    
    // R² (simulated)
    let r2_score = calculate_r_squared(&predictions, &targets)?;
    println!("  📊 R²: {:.4}", r2_score);
    
    // MAPE (simulated)
    let mape_score = calculate_mape(&predictions, &targets)?;
    println!("  📈 MAPE: {:.4}%", mape_score);
    
    Ok(())
}

async fn evaluate_ranking_metrics() -> GaussRDLResult<()> {
    // Sample ranking data (relevance scores)
    let predictions = vec![0.9, 0.7, 0.5, 0.3, 0.8, 0.2, 0.6, 0.4];
    let targets = vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    
    println!("Sample data: {} predictions vs {} targets", predictions.len(), targets.len());
    
    // MAP (Mean Average Precision)
    let map_score = calculate_map(&predictions, &targets, 5)?;
    println!("  🎯 MAP@5: {:.4}", map_score);
    
    // MRR (Mean Reciprocal Rank)
    let mrr_score = calculate_mrr(&predictions, &targets)?;
    println!("  🔝 MRR: {:.4}", mrr_score);
    
    // NDCG (Normalized Discounted Cumulative Gain)
    let ndcg_score = calculate_ndcg(&predictions, &targets, 5)?;
    println!("  📊 NDCG@5: {:.4}", ndcg_score);
    
    // Hits (Hit Rate) - simulated
    let hits_score = calculate_hits(&predictions, &targets, 3)?;
    println!("  🎯 Hits@3: {:.4}", hits_score);
    
    Ok(())
}

async fn evaluate_link_prediction_metrics() -> GaussRDLResult<()> {
    // Sample link prediction data
    let predictions = vec![0.85, 0.65, 0.45, 0.25, 0.75, 0.35, 0.55, 0.15];
    let targets = vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    
    println!("Sample data: {} predictions vs {} targets", predictions.len(), targets.len());
    
    // Link Prediction AUROC (simulated)
    let link_auroc_score = calculate_auroc(&predictions, &targets)?;
    println!("  🔗 Link Prediction AUROC: {:.4}", link_auroc_score);
    
    // Average Precision for Link Prediction
    let ap_score = calculate_average_precision(&predictions, &targets)?;
    println!("  📈 Average Precision: {:.4}", ap_score);
    
    Ok(())
}

async fn compare_metrics_performance() -> GaussRDLResult<()> {
    println!("Comparing performance of different metrics...");
    
    // Generate larger test data
    let size = 10000;
    let predictions: Vec<f64> = (0..size).map(|i| (i as f64 / size as f64) + 0.1 * ((i * 17) % 100) as f64 / 100.0).collect();
    let targets: Vec<f64> = (0..size).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect();
    
    println!("Test data size: {} samples", size);
    
    // Benchmark different metrics using helper functions
    let metrics = vec![
        ("AUROC", "auroc"),
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1Score", "f1"),
        ("RMSE", "rmse"),
        ("MAE", "mae"),
        ("MAP@10", "map"),
        ("MRR", "mrr"),
        ("NDCG@10", "ndcg"),
    ];
    
    for (name, metric_type) in metrics {
        let start_time = Instant::now();
        let score = match metric_type {
            "auroc" => calculate_auroc(&predictions, &targets)?,
            "accuracy" => calculate_accuracy(&predictions, &targets)?,
            "precision" => calculate_precision(&predictions, &targets)?,
            "recall" => calculate_recall(&predictions, &targets)?,
            "f1" => calculate_f1(&predictions, &targets)?,
            "rmse" => calculate_rmse(&predictions, &targets)?,
            "mae" => calculate_mae(&predictions, &targets)?,
            "map" => calculate_map(&predictions, &targets, 10)?,
            "mrr" => calculate_mrr(&predictions, &targets)?,
            "ndcg" => calculate_ndcg(&predictions, &targets, 10)?,
            _ => 0.0,
        };
        let eval_time = start_time.elapsed();
        
        println!("  📊 {}: {:.4} (evaluated in {:?})", name, score, eval_time);
    }
    
    Ok(())
}

// Helper functions to simulate metric calculations
fn calculate_auroc(predictions: &[f64], targets: &[f64]) -> GaussRDLResult<f64> {
    // Simplified AUROC calculation
    let mut pairs: Vec<(f64, f64)> = predictions.iter().zip(targets.iter()).map(|(&p, &t)| (p, t)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    
    let mut tp = 0.0;
    let mut _fp = 0.0;
    let pos_count = targets.iter().filter(|&&t| t > 0.5).count() as f64;
    let neg_count = targets.len() as f64 - pos_count;
    
    let mut auc = 0.0;
    for (_, target) in pairs {
        if target > 0.5 {
            tp += 1.0;
        } else {
            _fp += 1.0;
            auc += tp;
        }
    }
    
    if pos_count > 0.0 && neg_count > 0.0 {
        Ok(auc / (pos_count * neg_count))
    } else {
        Ok(0.5)
    }
}

fn calculate_accuracy(predictions: &[f64], targets: &[f64]) -> GaussRDLResult<f64> {
    if predictions.len() != targets.len() || predictions.is_empty() {
        return Ok(0.0);
    }
    
    let correct = predictions.iter().zip(targets.iter())
        .filter(|(&p, &t)| (p > 0.5) == (t > 0.5))
        .count();
    
    Ok(correct as f64 / predictions.len() as f64)
}

fn calculate_precision(predictions: &[f64], targets: &[f64]) -> GaussRDLResult<f64> {
    let tp = predictions.iter().zip(targets.iter())
        .filter(|(&p, &t)| p > 0.5 && t > 0.5)
        .count() as f64;
    
    let fp = predictions.iter().zip(targets.iter())
        .filter(|(&p, &t)| p > 0.5 && t <= 0.5)
        .count() as f64;
    
    if tp + fp > 0.0 {
        Ok(tp / (tp + fp))
    } else {
        Ok(0.0)
    }
}

fn calculate_recall(predictions: &[f64], targets: &[f64]) -> GaussRDLResult<f64> {
    let tp = predictions.iter().zip(targets.iter())
        .filter(|(&p, &t)| p > 0.5 && t > 0.5)
        .count() as f64;
    
    let fn_count = predictions.iter().zip(targets.iter())
        .filter(|(&p, &t)| p <= 0.5 && t > 0.5)
        .count() as f64;
    
    if tp + fn_count > 0.0 {
        Ok(tp / (tp + fn_count))
    } else {
        Ok(0.0)
    }
}

fn calculate_f1(predictions: &[f64], targets: &[f64]) -> GaussRDLResult<f64> {
    let precision = calculate_precision(predictions, targets)?;
    let recall = calculate_recall(predictions, targets)?;
    
    if precision + recall > 0.0 {
        Ok(2.0 * precision * recall / (precision + recall))
    } else {
        Ok(0.0)
    }
}

fn calculate_macro_f1(predictions: &[f64], targets: &[f64]) -> GaussRDLResult<f64> {
    // For binary case, macro F1 is the same as regular F1
    calculate_f1(predictions, targets)
}

fn calculate_rmse(predictions: &[f64], targets: &[f64]) -> GaussRDLResult<f64> {
    if predictions.len() != targets.len() || predictions.is_empty() {
        return Ok(0.0);
    }
    
    let mse = predictions.iter().zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>() / predictions.len() as f64;
    
    Ok(mse.sqrt())
}

fn calculate_mae(predictions: &[f64], targets: &[f64]) -> GaussRDLResult<f64> {
    if predictions.len() != targets.len() || predictions.is_empty() {
        return Ok(0.0);
    }
    
    let mae = predictions.iter().zip(targets.iter())
        .map(|(p, t)| (p - t).abs())
        .sum::<f64>() / predictions.len() as f64;
    
    Ok(mae)
}

fn calculate_r_squared(predictions: &[f64], targets: &[f64]) -> GaussRDLResult<f64> {
    if predictions.len() != targets.len() || predictions.is_empty() {
        return Ok(0.0);
    }
    
    let mean_target = targets.iter().sum::<f64>() / targets.len() as f64;
    let ss_tot = targets.iter().map(|&t| (t - mean_target).powi(2)).sum::<f64>();
    let ss_res = predictions.iter().zip(targets.iter()).map(|(&p, &t)| (t - p).powi(2)).sum::<f64>();
    
    if ss_tot > 0.0 {
        Ok(1.0 - (ss_res / ss_tot))
    } else {
        Ok(0.0)
    }
}

fn calculate_mape(predictions: &[f64], targets: &[f64]) -> GaussRDLResult<f64> {
    if predictions.len() != targets.len() || predictions.is_empty() {
        return Ok(0.0);
    }
    
    let mape = predictions.iter().zip(targets.iter())
        .filter(|(_, &t)| t.abs() > 1e-8) // Avoid division by zero
        .map(|(&p, &t)| ((t - p).abs() / t.abs()) * 100.0)
        .sum::<f64>();
    
    let valid_count = predictions.iter().zip(targets.iter())
        .filter(|(_, &t)| t.abs() > 1e-8)
        .count();
    
    if valid_count > 0 {
        Ok(mape / valid_count as f64)
    } else {
        Ok(0.0)
    }
}

fn calculate_map(predictions: &[f64], targets: &[f64], k: usize) -> GaussRDLResult<f64> {
    let mut pairs: Vec<(f64, f64)> = predictions.iter().zip(targets.iter()).map(|(&p, &t)| (p, t)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    
    let mut ap = 0.0;
    let mut relevant_found = 0.0;
    
    for (i, (_, target)) in pairs.iter().take(k).enumerate() {
        if *target > 0.5 {
            relevant_found += 1.0;
            ap += relevant_found / (i + 1) as f64;
        }
    }
    
    let total_relevant = targets.iter().filter(|&&t| t > 0.5).count() as f64;
    if total_relevant > 0.0 {
        Ok(ap / total_relevant)
    } else {
        Ok(0.0)
    }
}

fn calculate_mrr(predictions: &[f64], targets: &[f64]) -> GaussRDLResult<f64> {
    let mut pairs: Vec<(f64, f64)> = predictions.iter().zip(targets.iter()).map(|(&p, &t)| (p, t)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    
    for (i, (_, target)) in pairs.iter().enumerate() {
        if *target > 0.5 {
            return Ok(1.0 / (i + 1) as f64);
        }
    }
    
    Ok(0.0)
}

fn calculate_ndcg(predictions: &[f64], targets: &[f64], k: usize) -> GaussRDLResult<f64> {
    let mut pairs: Vec<(f64, f64)> = predictions.iter().zip(targets.iter()).map(|(&p, &t)| (p, t)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    
    let dcg = pairs.iter().take(k).enumerate()
        .map(|(i, (_, target))| target / (2.0_f64).log2() * (i + 2) as f64)
        .sum::<f64>();
    
    let mut sorted_targets = targets.to_vec();
    sorted_targets.sort_by(|a, b| b.partial_cmp(a).unwrap());
    
    let idcg = sorted_targets.iter().take(k).enumerate()
        .map(|(i, target)| target / (2.0_f64).log2() * (i + 2) as f64)
        .sum::<f64>();
    
    if idcg > 0.0 {
        Ok(dcg / idcg)
    } else {
        Ok(0.0)
    }
}

fn calculate_hits(predictions: &[f64], targets: &[f64], k: usize) -> GaussRDLResult<f64> {
    let mut pairs: Vec<(f64, f64)> = predictions.iter().zip(targets.iter()).map(|(&p, &t)| (p, t)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    
    let hits = pairs.iter().take(k).any(|(_, target)| *target > 0.5);
    Ok(if hits { 1.0 } else { 0.0 })
}

fn calculate_average_precision(predictions: &[f64], targets: &[f64]) -> GaussRDLResult<f64> {
    calculate_map(predictions, targets, predictions.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_auroc_metric() {
        let predictions = vec![0.9, 0.2, 0.8, 0.1];
        let targets = vec![1.0, 0.0, 1.0, 0.0];
        
        let score = calculate_auroc(&predictions, &targets).unwrap();
        assert!(score >= 0.0 && score <= 1.0);
    }
    
    #[test]
    fn test_rmse_metric() {
        let predictions = vec![2.5, 3.0, 1.5, 4.0];
        let targets = vec![2.3, 3.1, 1.7, 4.1];
        
        let score = calculate_rmse(&predictions, &targets).unwrap();
        assert!(score >= 0.0);
    }
    
    #[test]
    fn test_map_metric() {
        let predictions = vec![0.9, 0.7, 0.5, 0.3];
        let targets = vec![1.0, 1.0, 0.0, 0.0];
        
        let score = calculate_map(&predictions, &targets, 3).unwrap();
        assert!(score >= 0.0 && score <= 1.0);
    }
    
    #[test]
    fn test_metric_creation() {
        // Test that metric structs can be created (only available ones)
        let _auroc = AUROC::new();
        let _rmse = RMSE::new();
        let _mae = MAE::new();
        let _f1 = F1Score::new();
        let _macro_f1 = MacroF1::new(2);
        let _map = MAP::new(5);
        let _mrr = MRR::new();
        let _ndcg = NDCG::new(10);
        let _ap = AveragePrecision::new();
        let _accuracy = Accuracy::new();
    }
    
    #[test]
    fn test_empty_inputs() {
        let predictions = vec![];
        let targets = vec![];
        
        // Should handle empty inputs gracefully
        let result = calculate_auroc(&predictions, &targets);
        assert!(result.is_ok());
    }
} 