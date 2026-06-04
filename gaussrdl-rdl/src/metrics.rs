//! Evaluation metrics matching RelBench conventions: ROC-AUC for entity
//! classification, MAE/RMSE for entity regression, and MAP@k for
//! recommendation. All operate on plain CPU vectors.

/// Area under the ROC curve via the rank-statistic (Mann–Whitney U) formula.
/// Returns 0.5 if a class is absent.
pub fn roc_auc(preds: &[f32], labels: &[f32]) -> f32 {
    assert_eq!(preds.len(), labels.len());
    let n = preds.len();
    if n == 0 {
        return 0.5;
    }
    // Rank predictions (average ranks for ties).
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| preds[a].partial_cmp(&preds[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && (preds[idx[j + 1]] - preds[idx[i]]).abs() < 1e-12 {
            j += 1;
        }
        // average rank (1-based) for the tie group [i, j]
        let avg = ((i + j) as f64) / 2.0 + 1.0;
        for k in i..=j {
            ranks[idx[k]] = avg;
        }
        i = j + 1;
    }
    let n_pos: f64 = labels.iter().filter(|&&l| l > 0.5).count() as f64;
    let n_neg = n as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.5;
    }
    let sum_pos_ranks: f64 = (0..n).filter(|&i| labels[i] > 0.5).map(|i| ranks[i]).sum();
    ((sum_pos_ranks - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)) as f32
}

/// Accuracy at threshold 0.5 over sigmoid(logit) — but here we accept already
/// probability-like or logit scores and threshold at 0.
pub fn accuracy_from_logits(logits: &[f32], labels: &[f32]) -> f32 {
    let n = logits.len();
    if n == 0 {
        return 0.0;
    }
    let correct = (0..n)
        .filter(|&i| ((logits[i] > 0.0) as i32 as f32) == labels[i])
        .count();
    correct as f32 / n as f32
}

/// Mean absolute error.
pub fn mae(preds: &[f32], labels: &[f32]) -> f32 {
    let n = preds.len();
    if n == 0 {
        return 0.0;
    }
    preds.iter().zip(labels).map(|(p, y)| (p - y).abs()).sum::<f32>() / n as f32
}

/// Root mean squared error.
pub fn rmse(preds: &[f32], labels: &[f32]) -> f32 {
    let n = preds.len();
    if n == 0 {
        return 0.0;
    }
    (preds.iter().zip(labels).map(|(p, y)| (p - y) * (p - y)).sum::<f32>() / n as f32).sqrt()
}

/// Mean Average Precision @ k. `scores[i]` ranks candidates for query `i`;
/// `relevant[i]` is the set of relevant candidate indices.
pub fn map_at_k(scores: &[Vec<f32>], relevant: &[Vec<usize>], k: usize) -> f32 {
    let q = scores.len();
    if q == 0 {
        return 0.0;
    }
    let mut total = 0.0f32;
    for i in 0..q {
        let mut order: Vec<usize> = (0..scores[i].len()).collect();
        order.sort_by(|&a, &b| {
            scores[i][b].partial_cmp(&scores[i][a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        let rel: std::collections::HashSet<usize> = relevant[i].iter().copied().collect();
        if rel.is_empty() {
            continue;
        }
        let mut hits = 0.0f32;
        let mut ap = 0.0f32;
        for (rank, &cand) in order.iter().take(k).enumerate() {
            if rel.contains(&cand) {
                hits += 1.0;
                ap += hits / (rank as f32 + 1.0);
            }
        }
        let denom = (rel.len() as f32).min(k as f32);
        total += ap / denom;
    }
    total / q as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auc_perfect_and_random() {
        let preds = vec![0.9, 0.8, 0.2, 0.1];
        let labels = vec![1.0, 1.0, 0.0, 0.0];
        assert!((roc_auc(&preds, &labels) - 1.0).abs() < 1e-6);

        let preds2 = vec![0.1, 0.2, 0.8, 0.9];
        assert!((roc_auc(&preds2, &labels) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn map_basic() {
        let scores = vec![vec![0.1, 0.9, 0.4]];
        let relevant = vec![vec![1usize]];
        // top-1 is index 1 which is relevant -> AP = 1.0
        assert!((map_at_k(&scores, &relevant, 3) - 1.0).abs() < 1e-6);
    }
}
