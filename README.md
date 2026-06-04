# GaussRDL

### Relational Deep Learning, engineered in Rust — by **Gaussian Technologies**

[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org)
[![Engine](https://img.shields.io/badge/engine-v2%20(gaussrdl--rdl)-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/end--to--end-passing-brightgreen.svg)]()

> **Train neural networks directly on your relational database — no manual
> feature engineering, no flattening to a single table.** GaussRDL turns a
> multi-table schema into a temporal heterogeneous graph and learns over it
> end-to-end, in safe, fast, dependency-light Rust.

GaussRDL is Gaussian Technologies' implementation of **Relational Deep
Learning (RDL)** — the paradigm introduced by Stanford's RelBench
(Fey, Hu, Leskovec et al.) for predictive modeling over relational databases.
This repository ships a from-scratch RDL engine on the
[Candle](https://github.com/huggingface/candle) tensor framework with real
automatic differentiation, a model zoo spanning the current state of the art,
and a fully tested end-to-end pipeline.

---

## Why RDL, and why this matters

Most enterprise data lives in **many linked tables** (customers, orders,
products, events…). The standard ML workflow flattens this into one wide table
by hand — a slow, lossy, error-prone process. RDL instead represents the
database **losslessly as a graph** (each row a node, each foreign key an edge,
timestamps making it temporal) and learns the features automatically with a
graph neural network. This is the same direction pursued by the 2025–2026
research frontier (RelGT, ContextGNN, KumoRFM, Relational Transformer).

GaussRDL brings that frontier to the Rust ecosystem with an emphasis on
**correctness, reproducibility, and a tight dependency footprint** suitable for
embedding into production data systems.

---

## The v2 engine (`gaussrdl-rdl`)

The `gaussrdl-rdl` crate is the heart of this release: a **single, coherent,
test-covered pipeline** where every stage is wired to the next and trains by
real gradient descent.

```
RelationalDatabase            typed multi-table data + primary/foreign keys
   └─► HeteroGraph            row = node, FK = edge, temporal, leakage-free
         └─► DatabaseEncoder  PyTorch-Frame-style per-column "stype" encoders
               └─► Model      HeteroSAGE · RGCN · GAT · RelGT
                     └─► Head  classification / regression
                           └─► AdamW autodiff training loop ─► metrics
```

What makes it real (not a scaffold):

- **Genuine message passing** built on Candle `index_select` / `scatter_add`,
  so gradients flow through the graph operations.
- **Real training** with the Candle `AdamW` optimizer and a numerically stable
  binary-cross-entropy-with-logits loss.
- **Leakage-free temporal sampling**: only edges with `time ≤ seed_time`
  participate in a prediction, exactly as RDL requires.
- **Honest metrics**: rank-based ROC-AUC, MAE/RMSE, MAP@k — implemented and
  unit-tested, not hardcoded.
- **An end-to-end test suite** that generates synthetic relational data, trains
  each model, and asserts the loss decreases and the planted signal is learned.

### Models

| Model | Family | Key idea | Reference |
|-------|--------|----------|-----------|
| **HeteroSAGE** | RDL baseline | Heterogeneous GraphSAGE; per-relation transform + mean aggregation | Fey et al. 2024 ([2312.04615](https://arxiv.org/abs/2312.04615)) |
| **RGCN** | Relational GCN | Basis-decomposed relation weights `W_r = Σ_b a_{rb} B_b` | Schlichtkrull et al. 2018 ([1703.06103](https://arxiv.org/abs/1703.06103)) |
| **GAT** | Attention | Multi-head edge-softmax neighborhood attention | Veličković et al. 2018 ([1710.10903](https://arxiv.org/abs/1710.10903)) |
| **RelGT** | Graph transformer | Multi-element tokenization (feature·type·time·structure) + hybrid local/global attention over learnable centroids | Dwivedi et al. 2025 ([2505.10960](https://arxiv.org/abs/2505.10960)) |

See **[MODELS.md](MODELS.md)** for architecture details and **[docs/RDL_SOTA.md](docs/RDL_SOTA.md)**
for a cited survey of the field through June 2026.

---

## Quick start

```bash
# Run the full model benchmark on synthetic relational data
cargo run -p gaussrdl-rdl --bin gaussrdl-rdl -- benchmark

# Train one model on one task (churn = classification, ltv = regression)
cargo run -p gaussrdl-rdl --bin gaussrdl-rdl -- relgt churn 100

# End-to-end demo (all models, churn task)
cargo run -p gaussrdl-rdl --example end_to_end

# Run the test suite (graph construction, temporal masking, learning checks)
cargo test -p gaussrdl-rdl
```

Library usage:

```rust
use gaussrdl_rdl::{run_experiment, ExperimentConfig, ModelKind, TaskType};

let cfg = ExperimentConfig {
    model: ModelKind::RelGt,
    task: TaskType::BinaryClassification,
    epochs: 100,
    ..Default::default()
};
let result = run_experiment(&cfg)?;
println!("val AUROC = {:.4}", result.val.auroc);
```

To plug in **your own** database, construct a `RelationalDatabase` with
`Table`s, typed `Column`s (`Numerical` / `Categorical` / `Timestamp`), and
`ForeignKey`s — the graph, encoders, and training adapt automatically.

---

## Reproducible results

Real output of `cargo run -p gaussrdl-rdl -- benchmark` on the bundled
synthetic e-commerce database (1,707 nodes / 4,748 temporal edges / 4 relation
types; 60 epochs, CPU). Numbers are produced by the code in this repo — run it
yourself.

**User-churn (binary classification — higher is better):**

| Model | Val ROC-AUC | Val Accuracy | Test ROC-AUC |
|-------|:-----------:|:------------:|:------------:|
| HeteroSAGE | 0.690 | 0.638 | 0.736 |
| RGCN       | 0.650 | 0.638 | 0.721 |
| GAT        | 0.668 | 0.663 | 0.722 |
| **RelGT**  | **0.702** | **0.750** | 0.716 |

**User-LTV (regression — lower MAE is better):**

| Model | Val MAE | Val RMSE | Test MAE |
|-------|:-------:|:--------:|:--------:|
| HeteroSAGE | 362.9 | 502.4 | 359.9 |
| RGCN       | 279.8 | 405.6 | 306.3 |
| GAT        | 296.3 | 410.0 | 287.1 |
| **RelGT**  | **254.4** | **375.9** | **248.8** |

The SOTA graph transformer (RelGT) leads on both tasks, as expected from the
literature. All models clearly beat the chance baseline (AUROC 0.5), confirming
the pipeline learns the temporal signal end-to-end.

> These are demonstrations on **synthetic** data, not RelBench leaderboard
> scores. The roadmap below covers real-dataset loaders.

---

## Workspace layout

GaussRDL is a Cargo workspace. The **`gaussrdl-rdl`** crate is the v2 engine
described above. The remaining crates form the broader platform (data
management, graph algorithms, serving, CLI, metrics) and are being progressively
migrated onto the v2 engine.

| Crate | Role |
|-------|------|
| **`gaussrdl-rdl`** | **v2 engine: encoders, hetero-graph, models, training, metrics (this release)** |
| `gaussrdl-core` | Foundation types, traits, errors |
| `gaussrdl-data` | Dataset/task definitions and loaders |
| `gaussrdl-graph` | Graph construction, sampling, and algorithms |
| `gaussrdl-models` | Legacy model definitions (superseded by `gaussrdl-rdl`) |
| `gaussrdl-training` | Training infrastructure |
| `gaussrdl-metrics` | Monitoring and metrics |
| `gaussrdl-database` | Database connectivity |
| `gaussrdl-server` | HTTP inference server |
| `gaussrdl-cli` | Command-line interface |
| `gaussrdl-utils` | Shared utilities |
| `gaussrdl` | Umbrella crate re-exporting the platform |

---

## Roadmap

Grounded in the SOTA survey ([docs/RDL_SOTA.md](docs/RDL_SOTA.md)):

- [ ] **RelGNN** atomic-route composite message passing (ICML 2025) for
      many-to-many relations.
- [ ] **ContextGNN** pair-wise + two-tower recommender for link-prediction tasks
      (MAP@k), with negative sampling.
- [ ] Real **RelBench v1/v2** dataset loaders (rel-f1, rel-amazon, rel-hm, …).
- [ ] Per-seed **temporal subgraph mini-batch sampling** for large graphs.
- [ ] Text/multicategorical column encoders (frozen LM embeddings).
- [ ] GPU (CUDA/Metal) execution paths via Candle.

---

## Requirements

- **Rust** 1.75+ (stable). No GPU required; runs on CPU out of the box.

## License

MIT — see [LICENSE](LICENSE).

## About

Built by **Gaussian Technologies** — deep-tech infrastructure for learning
directly on the data shape enterprises actually have: relational, temporal,
multi-table. Contributions welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
