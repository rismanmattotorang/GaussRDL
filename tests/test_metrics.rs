use relbench::metrics::{
    Metric, MAE, MSE, RMSE, AUROC, AveragePrecision, Accuracy, F1Score,
};
use ndarray::{array, ArrayView1};

#[test]
fn test_mae() {
    let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_pred = array![1.1, 2.2, 2.8, 4.2, 4.9];
    
    let mae = MAE;
    let error = mae.compute(y_true.view(), y_pred.view()).unwrap();
    assert!((error - 0.16).abs() < 1e-6);
    assert!(!mae.higher_is_better());
}

#[test]
fn test_mse() {
    let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_pred = array![1.1, 2.2, 2.8, 4.2, 4.9];
    
    let mse = MSE;
    let error = mse.compute(y_true.view(), y_pred.view()).unwrap();
    assert!((error - 0.034).abs() < 1e-6);
    assert!(!mse.higher_is_better());
}

#[test]
fn test_rmse() {
    let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_pred = array![1.1, 2.2, 2.8, 4.2, 4.9];
    
    let rmse = RMSE;
    let error = rmse.compute(y_true.view(), y_pred.view()).unwrap();
    assert!((error - 0.1844).abs() < 1e-4);
    assert!(!rmse.higher_is_better());
}

#[test]
fn test_auroc() {
    let y_true = array![1.0, 0.0, 1.0, 0.0, 1.0];
    let y_pred = array![0.9, 0.1, 0.8, 0.2, 0.7];
    
    let auroc = AUROC;
    let score = auroc.compute(y_true.view(), y_pred.view()).unwrap();
    assert_eq!(score, 1.0); // Perfect ranking
    assert!(auroc.higher_is_better());
    
    // Random predictions
    let y_pred = array![0.5, 0.5, 0.5, 0.5, 0.5];
    let score = auroc.compute(y_true.view(), y_pred.view()).unwrap();
    assert!((score - 0.5).abs() < 1e-6); // Random performance
}

#[test]
fn test_average_precision() {
    let y_true = array![1.0, 0.0, 1.0, 0.0, 1.0];
    let y_pred = array![0.9, 0.1, 0.8, 0.2, 0.7];
    
    let ap = AveragePrecision;
    let score = ap.compute(y_true.view(), y_pred.view()).unwrap();
    assert_eq!(score, 1.0); // Perfect ranking
    assert!(ap.higher_is_better());
    
    // Random predictions
    let y_pred = array![0.5, 0.5, 0.5, 0.5, 0.5];
    let score = ap.compute(y_true.view(), y_pred.view()).unwrap();
    assert!((score - 0.6).abs() < 1e-6); // Random performance
}

#[test]
fn test_accuracy() {
    let y_true = array![1.0, 0.0, 1.0, 0.0, 1.0];
    let y_pred = array![0.9, 0.1, 0.8, 0.2, 0.7];
    
    let accuracy = Accuracy;
    let score = accuracy.compute(y_true.view(), y_pred.view()).unwrap();
    assert_eq!(score, 1.0); // Perfect classification
    assert!(accuracy.higher_is_better());
    
    // Random predictions
    let y_pred = array![0.5, 0.5, 0.5, 0.5, 0.5];
    let score = accuracy.compute(y_true.view(), y_pred.view()).unwrap();
    assert!((score - 0.6).abs() < 1e-6); // Random performance
}

#[test]
fn test_f1_score() {
    let y_true = array![1.0, 0.0, 1.0, 0.0, 1.0];
    let y_pred = array![0.9, 0.1, 0.8, 0.2, 0.7];
    
    let f1 = F1Score;
    let score = f1.compute(y_true.view(), y_pred.view()).unwrap();
    assert_eq!(score, 1.0); // Perfect classification
    assert!(f1.higher_is_better());
    
    // Random predictions
    let y_pred = array![0.5, 0.5, 0.5, 0.5, 0.5];
    let score = f1.compute(y_true.view(), y_pred.view()).unwrap();
    assert!(score >= 0.0 && score <= 1.0);
}

#[test]
fn test_edge_cases() {
    let metrics: Vec<Box<dyn Metric>> = vec![
        Box::new(MAE),
        Box::new(MSE),
        Box::new(RMSE),
        Box::new(AUROC),
        Box::new(AveragePrecision),
        Box::new(Accuracy),
        Box::new(F1Score),
    ];
    
    // Empty arrays
    let y_true = array![];
    let y_pred = array![];
    
    for metric in metrics.iter() {
        let result = metric.compute(y_true.view(), y_pred.view());
        assert!(result.is_err());
    }
    
    // Arrays with NaN
    let y_true = array![1.0, f32::NAN, 1.0];
    let y_pred = array![0.9, 0.8, 0.7];
    
    for metric in metrics.iter() {
        let result = metric.compute(y_true.view(), y_pred.view());
        assert!(result.is_err());
    }
    
    // Arrays with different lengths
    let y_true = array![1.0, 0.0, 1.0];
    let y_pred = array![0.9, 0.8];
    
    for metric in metrics.iter() {
        let result = metric.compute(y_true.view(), y_pred.view());
        assert!(result.is_err());
    }
}

#[test]
fn test_metric_names() {
    let metrics: Vec<Box<dyn Metric>> = vec![
        Box::new(MAE),
        Box::new(MSE),
        Box::new(RMSE),
        Box::new(AUROC),
        Box::new(AveragePrecision),
        Box::new(Accuracy),
        Box::new(F1Score),
    ];
    
    let expected_names = vec![
        "mae",
        "mse",
        "rmse",
        "auroc",
        "ap",
        "accuracy",
        "f1",
    ];
    
    for (metric, expected_name) in metrics.iter().zip(expected_names.iter()) {
        assert_eq!(metric.name(), *expected_name);
    }
}

#[test]
fn test_metric_descriptions() {
    let metrics: Vec<Box<dyn Metric>> = vec![
        Box::new(MAE),
        Box::new(MSE),
        Box::new(RMSE),
        Box::new(AUROC),
        Box::new(AveragePrecision),
        Box::new(Accuracy),
        Box::new(F1Score),
    ];
    
    for metric in metrics.iter() {
        assert!(!metric.description().is_empty());
    }
}

#[test]
fn test_metric_higher_is_better() {
    let regression_metrics: Vec<Box<dyn Metric>> = vec![
        Box::new(MAE),
        Box::new(MSE),
        Box::new(RMSE),
    ];
    
    let classification_metrics: Vec<Box<dyn Metric>> = vec![
        Box::new(AUROC),
        Box::new(AveragePrecision),
        Box::new(Accuracy),
        Box::new(F1Score),
    ];
    
    for metric in regression_metrics.iter() {
        assert!(!metric.higher_is_better());
    }
    
    for metric in classification_metrics.iter() {
        assert!(metric.higher_is_better());
    }
} 