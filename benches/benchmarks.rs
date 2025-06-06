use criterion::{black_box, criterion_group, criterion_main, Criterion};
use relbench::{
    get_dataset, get_task,
    metrics::{MAE, MSE, RMSE, AUROC, AveragePrecision, Accuracy, F1Score, Metric},
    utils::memory::{self, MemoryConfig},
    Result,
};
use ndarray::{array, Array1};
use std::time::Duration;

fn bench_dataset_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataset_loading");
    group.measurement_time(Duration::from_secs(10));
    
    group.bench_function("rel-amazon", |b| {
        b.iter(|| {
            let dataset = get_dataset(black_box("rel-amazon"), false).unwrap();
            black_box(dataset)
        })
    });
    
    group.finish();
}

fn bench_task_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("task_loading");
    group.measurement_time(Duration::from_secs(10));
    
    group.bench_function("rel-amazon/user-churn", |b| {
        b.iter(|| {
            let task = get_task(black_box("rel-amazon"), black_box("user-churn"), false).unwrap();
            black_box(task)
        })
    });
    
    group.finish();
}

fn bench_data_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_iteration");
    group.measurement_time(Duration::from_secs(10));
    
    let task = get_task("rel-amazon", "user-churn", false).unwrap();
    let train = task.get_train_table().unwrap();
    
    group.bench_function("sequential", |b| {
        b.iter(|| {
            for i in 0..train.len() {
                let features = black_box(train.get_features(i).unwrap());
                black_box(features);
            }
        })
    });
    
    group.bench_function("batched", |b| {
        b.iter(|| {
            let mut iter = task.get_train_iter(32).unwrap();
            while let Some(batch) = iter.next().unwrap() {
                black_box(batch);
            }
        })
    });
    
    group.finish();
}

fn bench_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("metrics");
    group.measurement_time(Duration::from_secs(10));
    
    let size = 10000;
    let y_true = Array1::linspace(0., 1., size);
    let y_pred = Array1::linspace(0.1, 0.9, size);
    
    let metrics: Vec<Box<dyn Metric>> = vec![
        Box::new(MAE),
        Box::new(MSE),
        Box::new(RMSE),
        Box::new(AUROC),
        Box::new(AveragePrecision),
        Box::new(Accuracy),
        Box::new(F1Score),
    ];
    
    for metric in metrics {
        group.bench_function(metric.name(), |b| {
            b.iter(|| {
                let score = metric.compute(
                    black_box(y_true.view()),
                    black_box(y_pred.view()),
                ).unwrap();
                black_box(score)
            })
        });
    }
    
    group.finish();
}

fn bench_memory_mapping(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_mapping");
    group.measurement_time(Duration::from_secs(10));
    
    // Configure memory settings
    let config = MemoryConfig {
        use_memory_mapping: true,
        cache_size: 1024 * 1024 * 1024, // 1GB
        prefetch_size: 1024 * 1024, // 1MB
    };
    memory::init(config).unwrap();
    
    let dataset = get_dataset("rel-amazon", false).unwrap();
    
    group.bench_function("standard_loading", |b| {
        b.iter(|| {
            let table = dataset.get_table(black_box("interactions")).unwrap();
            black_box(table)
        })
    });
    
    group.bench_function("memory_mapped_loading", |b| {
        b.iter(|| {
            let table = dataset.get_table_mapped(black_box("interactions")).unwrap();
            black_box(table)
        })
    });
    
    group.finish();
}

fn bench_parallel_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_processing");
    group.measurement_time(Duration::from_secs(10));
    
    let task = get_task("rel-amazon", "user-churn", false).unwrap();
    let train = task.get_train_table().unwrap();
    
    group.bench_function("sequential_features", |b| {
        b.iter(|| {
            let features: Vec<_> = (0..train.len())
                .map(|i| train.get_features(i).unwrap())
                .collect();
            black_box(features)
        })
    });
    
    group.bench_function("parallel_features", |b| {
        b.iter(|| {
            use rayon::prelude::*;
            let features: Vec<_> = (0..train.len())
                .into_par_iter()
                .map(|i| train.get_features(i).unwrap())
                .collect();
            black_box(features)
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_dataset_loading,
    bench_task_loading,
    bench_data_iteration,
    bench_metrics,
    bench_memory_mapping,
    bench_parallel_processing,
);
criterion_main!(benches); 