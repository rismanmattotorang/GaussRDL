# GaussRDL Models

This document describes the model zoo in the **`gaussrdl-rdl`** v2 engine. All
models are real, trainable [Candle](https://github.com/huggingface/candle)
modules implementing the [`NodeEncoder`] interface: given the `[N, in_dim]` node
feature matrix from the column encoders and the temporal graph connectivity,
they produce `[N, hidden]` node representations for a task head.

For the research context and citations, see
[docs/RDL_SOTA.md](docs/RDL_SOTA.md).

> The legacy `gaussrdl-models` crate contains earlier, partially-stubbed model
> definitions; they are superseded by the implementations below and retained
> only for API compatibility during migration.

## Shared configuration

```rust
ModelConfig {
    hidden_dim: 64,     // node representation width
    num_layers: 2,      // message-passing / transformer layers
    num_heads: 4,       // attention heads (GAT, RelGT)
    num_bases: 4,       // basis matrices (RGCN)
    num_centroids: 16,  // global centroids (RelGT)
}
```

Build any model through the factory:

```rust
let model = gaussrdl_rdl::models::build(
    ModelKind::RelGt, in_dim, num_relations, num_node_types, &cfg, vb)?;
```

---

## 1. HeteroSAGE — the RDL baseline

Heterogeneous GraphSAGE, the reference architecture from *Relational Deep
Learning* (Fey et al., 2024, [arXiv:2312.04615](https://arxiv.org/abs/2312.04615)).

Per relation type `r`, messages are transformed by a relation-specific weight
and mean-aggregated, then combined with a self transform:

```
h_i' = ReLU( W_self·h_i  +  Σ_r  mean_{j ∈ N_r(i)} W_r·h_j )
```

- Per-relation linear maps; mean aggregation via `scatter_mean`.
- Residual connections between layers.
- **Best for:** a strong, fast, well-understood default.

## 2. RGCN — Relational GCN with basis decomposition

Schlichtkrull et al., 2018 ([arXiv:1703.06103](https://arxiv.org/abs/1703.06103)).
Relation weights share a small set of `B` basis matrices:

```
W_r = Σ_{b=1}^{B} a_{r,b} · B_b
```

This keeps the parameter count manageable across the many relation types in a
relational schema and improves generalization. The basis composition is
implemented as a differentiable `coeff @ stack(B)` matmul, so both the bases and
the per-relation coefficients are learned.

- **Best for:** schemas with many relation types; parameter efficiency.

## 3. GAT — Graph Attention Network

Veličković et al., 2018 ([arXiv:1710.10903](https://arxiv.org/abs/1710.10903)).
Multi-head attention with a numerically stable **edge-softmax**:

```
e_ij = LeakyReLU( a_dst·(W h_i) + a_src·(W h_j) )
α_ij = softmax_j(e_ij)          # normalized over each node's neighbors
h_i' = ReLU( Σ_j α_ij · W h_j )
```

Edge-softmax is computed with a scatter-based normalization so attention is
correct over variable-size neighborhoods.

- **Best for:** when neighbor importance varies and interpretable weights help.

## 4. RelGT — Relational Graph Transformer

Dwivedi et al., 2025 ([arXiv:2505.10960](https://arxiv.org/abs/2505.10960)).
GaussRDL implements a full-graph variant of the two defining ideas:

**Multi-element tokenization** — each node's input token fuses:
- cell **features** (column-encoder output),
- node **type** embedding,
- **time** encoding, and
- a **structural** (degree) encoding (standing in for subgraph PE).

**Hybrid attention** per layer:
- **Local attention** — multi-head, edge-softmax attention over graph neighbors
  (`q·k/√d` scoring, value aggregation by `scatter_sum`).
- **Global attention** — multi-head attention from every node to `B` **learnable
  centroids** summarizing database-wide context (`softmax` over centroids).
- Local + global are summed and passed through a transformer **feed-forward
  block** with residual connections and layer norm.

The paper maintains centroids via EMA k-means; GaussRDL learns them directly by
backprop (a simpler, fully-differentiable variant). RelGT leads the bundled
benchmark on both classification and regression.

- **Best for:** capturing long-range and database-wide context; highest accuracy.

---

## Benchmark (synthetic data, reproducible)

Output of `cargo run -p gaussrdl-rdl -- benchmark` (1,707 nodes / 4,748 temporal
edges / 4 relations; 60 epochs; CPU). Run it to reproduce.

**User-churn (classification, ROC-AUC ↑):**

| Model | Val AUROC | Val Acc | Test AUROC |
|-------|:---------:|:-------:|:----------:|
| HeteroSAGE | 0.690 | 0.638 | 0.736 |
| RGCN | 0.650 | 0.638 | 0.721 |
| GAT | 0.668 | 0.663 | 0.722 |
| **RelGT** | **0.702** | **0.750** | 0.716 |

**User-LTV (regression, MAE ↓):**

| Model | Val MAE | Val RMSE | Test MAE |
|-------|:-------:|:--------:|:--------:|
| HeteroSAGE | 362.9 | 502.4 | 359.9 |
| RGCN | 279.8 | 405.6 | 306.3 |
| GAT | 296.3 | 410.0 | 287.1 |
| **RelGT** | **254.4** | **375.9** | **248.8** |

These are demonstrations on synthetic data — not RelBench leaderboard scores.

## Roadmap models

Planned, grounded in the SOTA survey: **RelGNN** (atomic-route composite message
passing, ICML 2025), **ContextGNN** (pair-wise + two-tower recommender, ICLR
2025), and a **Relational Transformer**-style cell-token model for zero-shot
transfer.
