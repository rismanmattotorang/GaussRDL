//! Catalog of benchmark datasets.
//!
//! Two kinds of entries:
//! * **Official** RelBench datasets (Robinson, Ranjan, Hu, … Leskovec, NeurIPS
//!   2024) — described with full metadata, tasks, metrics, and citations, and a
//!   canonical source URL. Downloading these requires the source host to be
//!   reachable (it is allow-listed in permissive environments; a clear error is
//!   returned otherwise).
//! * **Prepared** GaussRDL benchmarks — deterministic, RelBench-style synthetic
//!   relational databases that materialize locally with no network, so the full
//!   download → train → evaluate → report flow works offline.

use serde::{Deserialize, Serialize};

/// Task family, mapped onto the engine's task type + headline metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskKind {
    /// Entity classification — metric: ROC-AUC (higher is better).
    Classification,
    /// Entity regression — metric: MAE (lower is better).
    Regression,
}

impl TaskKind {
    pub fn metric_name(&self) -> &'static str {
        match self {
            TaskKind::Classification => "ROC-AUC",
            TaskKind::Regression => "MAE",
        }
    }
    pub fn higher_is_better(&self) -> bool {
        matches!(self, TaskKind::Classification)
    }
}

/// A predictive task defined on a dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskInfo {
    pub name: String,
    pub kind: TaskKind,
    /// Seed-table label column (used for prepared datasets that ship labels).
    pub label: String,
    pub description: String,
}

/// Size preset for a prepared dataset.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PreparedScale {
    pub num_users: usize,
    pub num_items: usize,
    pub avg_trans_per_user: f64,
}

/// Where a dataset comes from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DatasetSource {
    /// Materialized locally (no network).
    Prepared(PreparedScale),
    /// Downloaded as a set of files relative to `base_url`.
    Remote { base_url: String, files: Vec<String> },
}

/// Full description of a catalog dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    pub id: String,
    pub display_name: String,
    pub domain: String,
    pub description: String,
    pub citation: String,
    pub license: String,
    pub approx_size: String,
    pub seed_table: String,
    pub tables: Vec<String>,
    pub tasks: Vec<TaskInfo>,
    pub source: DatasetSource,
    /// True for real RelBench datasets; false for GaussRDL prepared benchmarks.
    pub official: bool,
}

impl DatasetInfo {
    pub fn task(&self, name: &str) -> Option<&TaskInfo> {
        self.tasks.iter().find(|t| t.name == name)
    }
    /// Whether this dataset can be obtained without external network access.
    pub fn offline_capable(&self) -> bool {
        matches!(self.source, DatasetSource::Prepared(_))
    }
}

fn ecommerce_tasks() -> Vec<TaskInfo> {
    vec![
        TaskInfo {
            name: "user-churn".into(),
            kind: TaskKind::Classification,
            label: "churn".into(),
            description: "Predict whether a user makes no purchase in the next window.".into(),
        },
        TaskInfo {
            name: "user-ltv".into(),
            kind: TaskKind::Regression,
            label: "ltv".into(),
            description: "Predict a user's future lifetime value (sum of purchases).".into(),
        },
    ]
}

/// The full dataset catalog.
pub fn catalog() -> Vec<DatasetInfo> {
    vec![
        // ---- Prepared (offline, runnable here) ----
        DatasetInfo {
            id: "gauss-ecom-small".into(),
            display_name: "Gauss e-commerce (small)".into(),
            domain: "e-commerce".into(),
            description: "RelBench-style users/items/transactions with a learnable temporal churn and LTV signal. Materializes locally; ideal for fast iteration.".into(),
            citation: "GaussRDL prepared benchmark (Gaussian Technologies, 2026).".into(),
            license: "MIT".into(),
            approx_size: "~0.5 MB".into(),
            seed_table: "users".into(),
            tables: vec!["users".into(), "items".into(), "transactions".into()],
            tasks: ecommerce_tasks(),
            source: DatasetSource::Prepared(PreparedScale { num_users: 400, num_items: 120, avg_trans_per_user: 12.0 }),
            official: false,
        },
        DatasetInfo {
            id: "gauss-ecom-medium".into(),
            display_name: "Gauss e-commerce (medium)".into(),
            domain: "e-commerce".into(),
            description: "A larger prepared e-commerce benchmark for more stable metric estimates.".into(),
            citation: "GaussRDL prepared benchmark (Gaussian Technologies, 2026).".into(),
            license: "MIT".into(),
            approx_size: "~4 MB".into(),
            seed_table: "users".into(),
            tables: vec!["users".into(), "items".into(), "transactions".into()],
            tasks: ecommerce_tasks(),
            source: DatasetSource::Prepared(PreparedScale { num_users: 2000, num_items: 500, avg_trans_per_user: 16.0 }),
            official: false,
        },
        // ---- Official RelBench (metadata + canonical source) ----
        official(
            "rel-f1", "RelBench: rel-f1", "sports (Formula 1)",
            "Formula 1 racing since 1950 (drivers, constructors, races, results).",
            vec![
                ("driver-dnf", TaskKind::Classification, "Will a driver not finish a race?"),
                ("driver-top3", TaskKind::Classification, "Will a driver finish in the top 3?"),
                ("driver-position", TaskKind::Regression, "Average finishing position."),
            ],
            "~74K rows · 9 tables",
        ),
        official(
            "rel-amazon", "RelBench: rel-amazon", "e-commerce",
            "Amazon product reviews (users, products, reviews).",
            vec![
                ("user-churn", TaskKind::Classification, "No review in the next 3 months."),
                ("item-churn", TaskKind::Classification, "No review of an item in the next 3 months."),
                ("user-ltv", TaskKind::Regression, "User lifetime value."),
                ("item-ltv", TaskKind::Regression, "Item lifetime value."),
            ],
            "~41M rows · 3 tables",
        ),
        official(
            "rel-hm", "RelBench: rel-hm", "e-commerce (fashion)",
            "H&M customer purchases (customers, articles, transactions).",
            vec![
                ("user-churn", TaskKind::Classification, "Customer churn."),
                ("item-sales", TaskKind::Regression, "Article sales."),
            ],
            "~30M rows · 3 tables",
        ),
        official(
            "rel-stack", "RelBench: rel-stack", "Q&A (Stack Exchange)",
            "Stack Exchange posts, users, votes, comments, badges.",
            vec![
                ("user-engagement", TaskKind::Classification, "Will a user be active?"),
                ("user-badge", TaskKind::Classification, "Will a user earn a badge?"),
                ("post-votes", TaskKind::Regression, "Votes a post receives."),
            ],
            "~media · 7 tables",
        ),
        official(
            "rel-trial", "RelBench: rel-trial", "clinical (ClinicalTrials.gov)",
            "Clinical trials, conditions, interventions, outcomes.",
            vec![
                ("study-outcome", TaskKind::Classification, "Will a study meet its outcome?"),
                ("study-adverse", TaskKind::Regression, "Number of adverse events."),
            ],
            "~media · 15 tables",
        ),
        official(
            "rel-avito", "RelBench: rel-avito", "classifieds",
            "Avito classified ads (users, ads, searches, clicks).",
            vec![
                ("user-clicks", TaskKind::Classification, "Will a user click an ad?"),
                ("ad-ctr", TaskKind::Regression, "Ad click-through rate."),
            ],
            "~media · 8 tables",
        ),
        official(
            "rel-event", "RelBench: rel-event", "social/events",
            "Hangtime event app (users, events, friends, attendance).",
            vec![
                ("user-repeat", TaskKind::Classification, "Will a user attend again?"),
                ("user-ignore", TaskKind::Classification, "Will a user ignore invitations?"),
            ],
            "~media · 5 tables",
        ),
    ]
}

/// Build an official RelBench catalog entry.
fn official(
    id: &str,
    display: &str,
    domain: &str,
    description: &str,
    tasks: Vec<(&str, TaskKind, &str)>,
    size: &str,
) -> DatasetInfo {
    let base = format!("https://relbench.stanford.edu/download/{id}/");
    DatasetInfo {
        id: id.into(),
        display_name: display.into(),
        domain: domain.into(),
        description: description.into(),
        citation: "Robinson, Ranjan, Hu, Yuan, Bain, Fey, Leskovec et al., \"RelBench\", NeurIPS 2024 (arXiv:2407.20060).".into(),
        license: "See relbench.stanford.edu (per-dataset upstream licenses).".into(),
        approx_size: size.into(),
        seed_table: "".into(),
        tables: Vec::new(),
        tasks: tasks
            .into_iter()
            .map(|(n, k, d)| TaskInfo { name: n.into(), kind: k, label: String::new(), description: d.into() })
            .collect(),
        source: DatasetSource::Remote { base_url: base, files: vec!["schema.json".into()] },
        official: true,
    }
}

/// Look up a dataset by id.
pub fn find(id: &str) -> Option<DatasetInfo> {
    catalog().into_iter().find(|d| d.id == id)
}
