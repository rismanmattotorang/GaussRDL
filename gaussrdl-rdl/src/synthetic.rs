//! Synthetic RelBench-style relational data generator.
//!
//! Produces a small e-commerce database (`users`, `items`, `transactions`)
//! with a built-in, *learnable* temporal signal so the end-to-end pipeline can
//! be exercised and tested without downloading real datasets.
//!
//! Each user has a latent activity rate that governs both their past and
//! future purchasing. Because the rate persists across the seed time, a model
//! that reads a user's recent transaction neighborhood can predict future
//! churn / lifetime value — exactly the structure RDL is designed to exploit.

use crate::data::{Column, ColumnData, ForeignKey, RelationalDatabase, Table};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Configuration for the synthetic generator.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SyntheticConfig {
    pub num_users: usize,
    pub num_items: usize,
    pub num_countries: usize,
    pub num_categories: usize,
    /// Average number of transactions per user over the full time span.
    pub avg_trans_per_user: f64,
    /// Total time span in seconds.
    pub time_span: f64,
    /// Fraction of the span used as the seed time (features use t <= seed).
    pub seed_fraction: f64,
    /// Future window (as a fraction of the span) used to derive labels.
    pub label_horizon_fraction: f64,
    pub seed: u64,
}

impl Default for SyntheticConfig {
    fn default() -> Self {
        Self {
            num_users: 400,
            num_items: 120,
            num_countries: 6,
            num_categories: 8,
            avg_trans_per_user: 12.0,
            time_span: 1000.0,
            seed_fraction: 0.7,
            label_horizon_fraction: 0.3,
            seed: 42,
        }
    }
}

/// A generated dataset: the database plus per-user supervised targets and the
/// seed time that separates feature history from label horizon.
pub struct SyntheticDataset {
    pub db: RelationalDatabase,
    pub seed_time: f64,
    /// Binary churn label per user (1 = no purchase in the label window).
    pub churn_labels: Vec<f32>,
    /// Future lifetime-value (sum of future purchase amounts) per user.
    pub ltv_labels: Vec<f32>,
    pub config: SyntheticConfig,
}

impl SyntheticDataset {
    /// Generate a dataset from `cfg`.
    pub fn generate(cfg: SyntheticConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let seed_time = cfg.time_span * cfg.seed_fraction;
        let label_end = seed_time + cfg.time_span * cfg.label_horizon_fraction;

        // --- users table ---
        let mut user_age = Vec::with_capacity(cfg.num_users);
        let mut user_country = Vec::with_capacity(cfg.num_users);
        let mut user_signup = Vec::with_capacity(cfg.num_users);
        // Latent per-user activity rate (transactions / unit time).
        let mut activity = Vec::with_capacity(cfg.num_users);
        for _ in 0..cfg.num_users {
            user_age.push(rng.gen_range(18.0..70.0_f32));
            user_country.push(rng.gen_range(0..cfg.num_countries) as i64);
            user_signup.push(rng.gen_range(0.0..seed_time * 0.5));
            // Heavier tail: many low-activity users, few power users.
            let a: f64 = rng.gen_range(0.0..1.0_f64).powf(2.0);
            activity.push(a * cfg.avg_trans_per_user / cfg.time_span);
        }

        // --- items table ---
        let mut item_price = Vec::with_capacity(cfg.num_items);
        let mut item_category = Vec::with_capacity(cfg.num_items);
        let mut item_listed = Vec::with_capacity(cfg.num_items);
        for _ in 0..cfg.num_items {
            item_price.push(rng.gen_range(5.0..500.0_f32));
            item_category.push(rng.gen_range(0..cfg.num_categories) as i64);
            item_listed.push(rng.gen_range(0.0..seed_time * 0.5));
        }

        // --- transactions (PK-FK edges) ---
        let mut t_user: Vec<i64> = Vec::new();
        let mut t_item: Vec<i64> = Vec::new();
        let mut t_amount: Vec<f32> = Vec::new();
        let mut t_time: Vec<f64> = Vec::new();

        let mut churn = vec![1.0f32; cfg.num_users];
        let mut ltv = vec![0.0f32; cfg.num_users];

        for u in 0..cfg.num_users {
            // Expected count over the whole span; sample a concrete count.
            let lambda = activity[u] * cfg.time_span;
            let count = poisson(&mut rng, lambda);
            for _ in 0..count {
                let time = rng.gen_range(0.0..label_end);
                let item = rng.gen_range(0..cfg.num_items);
                let amount = item_price[item] * rng.gen_range(0.5..1.5_f32);
                if time <= seed_time {
                    // Historical transaction -> becomes a feature edge.
                    t_user.push(u as i64);
                    t_item.push(item as i64);
                    t_amount.push(amount);
                    t_time.push(time);
                } else if time <= label_end {
                    // Future transaction -> contributes to labels only.
                    churn[u] = 0.0;
                    ltv[u] += amount;
                }
            }
        }

        let num_trans = t_user.len();

        let users = Table::new(
            "users",
            cfg.num_users,
            vec![
                Column { name: "age".into(), data: ColumnData::Numerical(user_age) },
                Column {
                    name: "country".into(),
                    data: ColumnData::Categorical {
                        values: user_country,
                        cardinality: cfg.num_countries,
                    },
                },
                Column { name: "signup".into(), data: ColumnData::Timestamp(user_signup.iter().map(|&x| x as f64).collect()) },
            ],
            Some(user_signup.iter().map(|&x| x as f64).collect()),
        )
        .expect("users table");

        let items = Table::new(
            "items",
            cfg.num_items,
            vec![
                Column { name: "price".into(), data: ColumnData::Numerical(item_price) },
                Column {
                    name: "category".into(),
                    data: ColumnData::Categorical {
                        values: item_category,
                        cardinality: cfg.num_categories,
                    },
                },
            ],
            Some(item_listed.clone()),
        )
        .expect("items table");

        let transactions = Table::new(
            "transactions",
            num_trans,
            vec![
                Column {
                    name: "user_id".into(),
                    data: ColumnData::Categorical { values: t_user, cardinality: cfg.num_users },
                },
                Column {
                    name: "item_id".into(),
                    data: ColumnData::Categorical { values: t_item, cardinality: cfg.num_items },
                },
                Column { name: "amount".into(), data: ColumnData::Numerical(t_amount) },
            ],
            Some(t_time),
        )
        .expect("transactions table");

        let mut db = RelationalDatabase::new("synthetic-ecommerce");
        db.add_table(users).add_table(items).add_table(transactions);
        db.add_foreign_key(ForeignKey {
            src_table: "transactions".into(),
            column: "user_id".into(),
            dst_table: "users".into(),
        });
        db.add_foreign_key(ForeignKey {
            src_table: "transactions".into(),
            column: "item_id".into(),
            dst_table: "items".into(),
        });

        SyntheticDataset {
            db,
            seed_time,
            churn_labels: churn,
            ltv_labels: ltv,
            config: cfg,
        }
    }
}

/// Knuth's algorithm for sampling a Poisson(`lambda`) count.
fn poisson(rng: &mut StdRng, lambda: f64) -> usize {
    if lambda <= 0.0 {
        return 0;
    }
    let l = (-lambda).exp();
    let mut k = 0usize;
    let mut p = 1.0f64;
    loop {
        k += 1;
        p *= rng.gen_range(0.0..1.0_f64);
        if p <= l {
            return k - 1;
        }
        if k > 10_000 {
            return k; // safety valve
        }
    }
}
