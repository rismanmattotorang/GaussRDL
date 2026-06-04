# GaussRDL Model Implementation Review

An honest assessment of how faithfully each `gaussrdl-rdl` model implements its
source architecture, what is simplified, and the quality/regularization
improvements applied in this release. Citations are in
[docs/RDL_SOTA.md](RDL_SOTA.md).

## Methodology

Each model is exercised by the integration suite, which asserts the training
loss decreases and held-out **val ROC-AUC > 0.6** (classification) on synthetic
data with a planted temporal signal. Gradients are verified to flow through
every parameter (the basis coefficients, attention vectors, and centroids are
all trainable — confirmed by the loss decreasing and by `grad_norm > 0` each
epoch, which the trainer records).

## Cross-cutting quality improvements (this release)

| Area | Before | Now |
|------|--------|-----|
| Regularization | none | train-gated **dropout** between layers |
| Optimization | fixed LR | **cosine / warmup-cosine** LR schedule |
| Selection | last epoch | **early stopping** + best-checkpoint restore |
| Monitoring | final only | per-epoch loss, val metric, **gradient norm**, LR, time |
| Persistence | none | VarMap **checkpoint save/load** (safetensors) |
| Numerical stability | `sigmoid(x).log()` (NaN-prone) | stable `relu(x) − x·z + softplus(−|x|)` BCE |

## Per-model assessment

### HeteroSAGE — RDL baseline · **faithful**
- **Implements:** per-relation linear message transforms with **mean
  aggregation** and a self transform, residual stacking — i.e. the heterogeneous
  GraphSAGE used as the RelBench baseline.
- **Simplifications:** full-graph (not neighbor-sampled) message passing; single
  aggregation function (mean). Adequate for the graph sizes targeted here.
- **Quality:** correct gather/scatter with gradient flow; dropout + residuals.
- **Gaps / roadmap:** per-seed neighbor sampling for scale; max/sum/attention
  aggregators.

### RGCN — basis decomposition · **faithful**
- **Implements:** `W_r = Σ_b a_{r,b} B_b`, composed **differentiably** as
  `coeff @ stack(B)` so both bases and coefficients learn. (The first draft
  read coefficients out to scalars, which would have **blocked gradients** to
  `coeff`; this was fixed — see `models/rgcn.rs`.)
- **Simplifications:** mean (degree-normalized) aggregation rather than the
  original normalized-sum; no block-diagonal decomposition variant.
- **Quality:** parameter-efficient; correct relation masking.
- **Gaps / roadmap:** block-diagonal decomposition; per-relation
  normalization choices.

### GAT — multi-head attention · **faithful**
- **Implements:** the GAT scoring `LeakyReLU(a_dst·Wh_i + a_src·Wh_j)` with a
  **scatter-based edge-softmax** that normalizes correctly over each node's
  variable-size neighborhood, multi-head value aggregation.
- **Simplifications:** relation-agnostic (homogeneous) attention — a deliberate
  contrast to the relation-aware models; global max-subtraction for softmax
  stability rather than per-destination max.
- **Quality:** numerically stable softmax; heads validated against `hidden_dim`.
- **Gaps / roadmap:** per-relation attention; GATv2 dynamic attention.

### RelGT — Relational Graph Transformer · **faithful in spirit, simplified**
- **Implements the two defining ideas** of the paper:
  1. **Multi-element tokenization** — token = cell features + node-type
     embedding + time encoding + structural (degree) encoding.
  2. **Hybrid attention** — multi-head **local** edge-softmax attention over
     graph neighbors **plus global** multi-head attention to `B` **learnable
     centroids**, fused through a transformer FFN block with residuals + layer
     norm.
- **Simplifications vs. the paper:**
  - Centroids are learned by **backprop** rather than maintained by **EMA
    k-means** over mini-batch seed features.
  - Structural token uses **node degree** as a stand-in for the subgraph
    positional encoding; **hop** distance is not separately tokenized in the
    full-graph variant.
  - Operates **full-graph** rather than over per-seed temporally-sampled
    subgraphs.
- **Quality:** real `softmax` attention (the legacy crate had skipped softmax);
  best performer on the bundled benchmark for both tasks.
- **Gaps / roadmap:** EMA-k-means centroids, explicit hop tokens, Laplacian/RWPE
  subgraph PE, per-seed subgraph batching.

## Working with real data — readiness

- **Implemented:** CSV ingestion (`io.rs`) with a JSON schema (numerical /
  categorical / timestamp / PK / FK / label roles), foreign-key resolution to
  row indices, categorical interning, timestamp parsing (epoch seconds or
  `YYYY-MM-DD`), and a CSV round-trip test that trains end-to-end from files.
- **Leakage control:** temporal edge masking at `t ≤ seed_time`, with the seed
  time chosen as a configurable quantile of observed event times.
- **Key gaps for production RelBench-scale data:**
  - **Parquet / database** connectors (only CSV today).
  - **Mini-batch temporal subgraph sampling** (current path is full-graph, so
    memory grows with the database).
  - **Text / multicategorical** column encoders (frozen LM embeddings).
  - **Recommendation/link-prediction** tasks (MAP@k) — metric implemented;
    ContextGNN model on the roadmap.
  - **GPU** execution (CPU only today).

## Summary

The four models are correct, trainable, and regularized implementations of
their respective architectures, with RelGT and RGCN/GAT simplified in clearly
documented ways. The most important correctness fix was making RGCN's basis
composition differentiable. The largest remaining gap for *real, large* data is
mini-batch temporal subgraph sampling and richer column encoders.
