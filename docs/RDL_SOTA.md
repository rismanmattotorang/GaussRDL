# State of the Art in Relational Deep Learning (through June 2026)

This survey grounds GaussRDL's design in the published literature. Relational
Deep Learning (RDL) trains neural networks directly on multi-table relational
databases by representing the database as a heterogeneous, temporal graph and
learning over it end-to-end, eliminating manual feature engineering.

> Sourcing note: where a primary PDF could not be fetched directly, claims are
> drawn from author abstracts, official GitHub READMEs, and reputable
> secondary summaries. Exact layer/head counts for some 2025 models are not
> reproduced here; consult the cited repos for configs.

## 1. The RelBench benchmark

**RelBench v1** — Robinson, Ranjan, Hu, Yuan, Bain, Fey, Leskovec et al.,
*"RelBench: A Benchmark for Deep Learning on Relational Databases,"* NeurIPS
2024 Datasets & Benchmarks ([arXiv:2407.20060](https://arxiv.org/abs/2407.20060),
[relbench.stanford.edu](https://relbench.stanford.edu/)). Ships 7 databases and
30 tasks: e-commerce (**rel-amazon**, **rel-hm**, **rel-avito**), Q&A
(**rel-stack**), events (**rel-event**), clinical (**rel-trial**), and sports
(**rel-f1**). Three task types and their metrics:

- **Entity classification** — binary node label at a seed time; metric **ROC-AUC**.
- **Entity regression** — numerical node label; metric **MAE**.
- **Recommendation / link prediction** — top-K targets per source; metric **MAP@K**.

Each task has a fixed **seed time** and time-based train/val/test splits.

**RelBench v2** (preprint ~Feb 2026) adds databases (rel-salt, rel-ratebeer,
rel-arxiv, rel-mimic) and 36 tasks, including the **Autocomplete** paradigm
(predict an existing column's value at a timestamp from relational + temporal
context) and integration with the Temporal Graph Benchmark (TGB).

## 2. The RDL baseline — heterogeneous GNN + column encoders

Position paper: Fey, Hu, Huang, Lenssen, Ranjan, Robinson, Ying, You, Leskovec,
*"Relational Deep Learning: Graph Representation Learning on Relational
Databases,"* 2024 ([arXiv:2312.04615](https://arxiv.org/abs/2312.04615)).
Pipeline:

1. **DB → graph.** Each row → a node typed by its table; each primary-key →
   foreign-key reference → a typed edge; row timestamps make the graph temporal.
2. **Cell → features.** Encode each row with **PyTorch Frame**
   ([arXiv:2404.00776](https://arxiv.org/abs/2404.00776)) per-column **semantic
   type ("stype")** encoders (numerical, categorical, multicategorical,
   text_embedded, text_tokenized, timestamp, …), producing `[rows, num_cols,
   channels]`, then a column-interaction model (ResNet/FT-Transformer/Trompt)
   fuses columns into one node embedding.
3. **Message passing.** A **heterogeneous GraphSAGE** GNN aggregates over typed
   neighbors:
   `h_i' = σ(W_self·h_i + Σ_r AGG_{j∈N_r(i)} W_r·h_j)`. A small MLP head on the
   seed node's final embedding produces the prediction.
4. **Temporal neighbor sampling** prevents leakage.

Consolidating survey: Leskovec et al., KDD 2025
([arXiv:2506.16654](https://arxiv.org/abs/2506.16654)).

## 3. Column ("stype") encoders

The stype abstraction drives RDL's input layer. Each column is embedded by an
stype-specific module — numerical (affine/linear), categorical (embedding
table), multicategorical (pooled embeddings), timestamp (cyclic/periodic
features), text (frozen or trainable LM) — then summed/fused per row.
**RELATE** ([arXiv:2510.19954](https://arxiv.org/abs/2510.19954), 2025) proposes
shared modality modules + Perceiver-style cross-attention for schema-agnostic,
permutation-invariant per-node encodings.

## 4. Temporal neighbor sampling

To avoid label leakage, RDL uses **time-consistent sampling**: for a seed entity
at seed time *t*, the sampled subgraph contains only nodes/edges with timestamp
`≤ t`. Sampling is done on-the-fly over one shared entity graph. The same `≤ t`
constraint is reused by RelGT, RelGNN, ContextGNN, and the Autocomplete tasks.

## 5. SOTA architectures (2024–2026)

### RelGNN — composite message passing (ICML 2025)
Chen, Kanatsoulis, Leskovec ([arXiv:2502.06784](https://arxiv.org/abs/2502.06784),
[code](https://github.com/snap-stanford/RelGNN)). Defines **atomic routes** —
simple paths enabling direct single-hop interaction between a source and
destination otherwise separated by a bridge node — and **composite message
passing + graph attention** along them. Reports SOTA on the large majority of
the 30 RelBench tasks (up to ~25% improvement).

### RelGT — Relational Graph Transformer (2025)  *(implemented in GaussRDL)*
Dwivedi, Jaladi, Shen, Lopez, Kanatsoulis, Puri, Fey, Leskovec
([arXiv:2505.10960](https://arxiv.org/abs/2505.10960),
[code](https://github.com/snap-stanford/relgt)). Two defining ideas:
- **Multi-element tokenization**: each node → five tokens — cell *features*,
  node *type*, *hop* distance, *time* (relative to seed), and *local
  structure / subgraph PE* — encoding heterogeneity, temporality, and topology
  without expensive precomputation.
- **Hybrid attention**: **local attention** over the temporally-sampled
  subgraph + **global attention to B learnable centroids**
  (`h_global = Attention(v, {c_b})`); the final embedding fuses both. Centroids
  are maintained by EMA k-means over mini-batch seed features.
  Reports matching/beating GNN baselines by up to ~18% across 21 tasks.

GaussRDL implements RelGT as a full-graph variant: multi-element token init
(feature + type + time + degree/structure), per-layer **multi-head local
edge-softmax attention** plus **global attention to learnable centroids**, fused
through a transformer FFN block. Centroids are learned by backprop (a simpler,
fully-differentiable alternative to EMA k-means).

### ContextGNN — beyond two-tower recommendation (ICLR 2025)
Yuan, Fey, Robinson et al.
([arXiv:2411.19513](https://arxiv.org/abs/2411.19513)). Single-stage hybrid:
a **pair-wise branch** (GNN over the user's k-hop subgraph for familiar items)
+ a **two-tower branch** (shallow item embeddings for exploratory items),
fused by a user-specific personalized score. Reports large gains over pure
pair-wise and two-tower baselines on RelBench recommendation.

## 6. Tabular & relational foundation models

- **TabPFN v2** (Hollmann et al., *Nature* 2025) — prior-data-fitted transformer
  doing in-context learning for single-table tasks; relevant as a per-table
  cell predictor.
- **CARTE** ([arXiv:2402.16785](https://arxiv.org/abs/2402.16785)) — row-as-star-
  graph with open-vocabulary string embeddings + graph-attention transformer.
- **Griffin** — early unified relational FM (temporal hetero-graph + table
  encoders + GNN).
- **KumoRFM** (Kumo.ai 2025) and **KumoRFM-2** (2026) — pretrained Relational
  Foundation Models doing in-context learning across arbitrary schemas via a
  Relational Graph Transformer over dynamically sampled context/prediction
  subgraphs.
- **Relational Transformer (RT)** — *"Toward Zero-Shot Foundation Models for
  Relational Data"* ([arXiv:2510.06377](https://arxiv.org/abs/2510.06377), ICLR
  2026): cells = tokens (trainable datatype encoding + frozen LM name
  embeddings), **relational attention** (a cell attends to its column and its
  row + FK-linked rows), masked-token pretraining for zero-shot transfer.

## 7. How GaussRDL maps to the SOTA

| Concept | GaussRDL component |
|---------|--------------------|
| DB → temporal hetero-graph | `graph::HeteroGraph` |
| Stype column encoders + fusion | `encoder::DatabaseEncoder` |
| Leakage-free temporal sampling | `HeteroGraph::edge_tensors(seed_time)` |
| RDL GNN baseline | `models::HeteroSAGE` |
| Relational GCN (basis) | `models::Rgcn` |
| Graph attention | `models::Gat` |
| Relational Graph Transformer | `models::RelGt` |
| ROC-AUC / MAE / MAP@k | `metrics` |
| Autodiff training (AdamW) | `pipeline::run_experiment` |

## Key sources
- RelBench v1: [arXiv:2407.20060](https://arxiv.org/abs/2407.20060) · [relbench.stanford.edu](https://relbench.stanford.edu/)
- RDL foundations: [arXiv:2312.04615](https://arxiv.org/abs/2312.04615) · survey [arXiv:2506.16654](https://arxiv.org/abs/2506.16654)
- PyTorch Frame: [arXiv:2404.00776](https://arxiv.org/abs/2404.00776)
- RelGNN: [arXiv:2502.06784](https://arxiv.org/abs/2502.06784) · RelGT: [arXiv:2505.10960](https://arxiv.org/abs/2505.10960) · ContextGNN: [arXiv:2411.19513](https://arxiv.org/abs/2411.19513)
- RELATE: [arXiv:2510.19954](https://arxiv.org/abs/2510.19954) · Relational Transformer: [arXiv:2510.06377](https://arxiv.org/abs/2510.06377)
- CARTE: [arXiv:2402.16785](https://arxiv.org/abs/2402.16785)
