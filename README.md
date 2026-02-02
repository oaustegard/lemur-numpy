# LEMUR-NumPy: CPU-Only Multi-Vector Retrieval

A PyTorch-free implementation of [LEMUR](https://arxiv.org/abs/2601.21853) (Learned Multi-Vector Retrieval). Forked from https://github.com/ejaasaari/lemur  

## CAVEAT: You should probably look at the original paper and source code  
This forked repo should be considered more a maybe-worthy-of-a-high-school-science-fair than anything approaching science. A learning exercise for me (the human, Oskar) about ColBERT, bi-encoders vs cross-encoders, etc., and a test of the refactoring-capabilities of Opus and jules.google.com

## Before You Start: Do You Need This? (Probably not)

**LEMUR is NOT an embedding model.** It's a retrieval optimization layer for ColBERT-style multi-vector embeddings. Before adopting this pipeline, understand what you're signing up for:

```
Query text 
    → [ColBERT model - YOU STILL NEED THIS] 
    → Multi-vector representation (N tokens × d dims)
    → [LEMUR MLP - this repo] 
    → Single vector (d' dims)
    → [HNSW/pgvector lookup] 
    → Results
```

If you're still in, go back to the original code and paper (see above).

**The GPU cost you care about is embedding, not retrieval.** LEMUR being "CPU-optimized" doesn't help GPU-poor setups—all vector retrieval (pgvector, Faiss, etc.) is CPU-based anyway. The GPU question is where you run ColBERT, which LEMUR doesn't replace.

### When LEMUR Makes Sense

- You've **already validated** that ColBERT outperforms single-vector models on your corpus
- You need ColBERT-quality retrieval as a **first stage** over millions of documents
- You're willing to maintain the ColBERT → LEMUR → rerank pipeline

### When to Skip LEMUR Entirely

- You're GPU-poor and haven't committed to ColBERT yet → **use hosted embedding APIs (OpenAI, Voyage, Jina) + pgvector**
- You're using ColBERT only for **reranking** top-100 candidates → MaxSim over 100 docs is already fast, LEMUR adds nothing
- You haven't benchmarked whether ColBERT actually beats BGE/E5/GTE on your data → **the gap is now 1-3 nDCG points, LEMUR's 5% approximation loss may erase it**

---

## What This Repo Provides

A PyTorch-free inference implementation of LEMUR's MLP projection.

| Metric | Original (PyTorch + C++) | NumPy + Numba |
|--------|--------------------------|---------------|
| QPS (10k docs) | ~6900 | ~660 |
| Performance | 100% | ~10% |
| Dependencies | PyTorch (190MB), C++ ext | NumPy, Numba (~5MB) |
| Training | Yes | **No** |
| Embedding | **No** | **No** |

**Critical limitations:**
- Training still requires PyTorch (use original repo, then export weights)
- You still need ColBERT (or similar) to embed queries at runtime
- This only helps the HNSW lookup step, which is rarely your bottleneck

## Installation

```bash
# Fetch repo
curl -sL https://api.github.com/repos/oaustegard/lemur-numpy/tarball/main | tar xz
cd oaustegard-lemur-numpy-*

# Install
pip install numba --break-system-packages
pip install -e . --break-system-packages
```

## Usage

### Prerequisites

You need:
1. **Pre-trained LEMUR weights** (exported from PyTorch version)
2. **ColBERT embeddings** for your corpus (precomputed)
3. **A way to embed queries** at runtime (ColBERT model, API, etc.)

### Inference

```python
from lemur.lemur_numpy import LemurNumPy
import numpy as np

# Load pre-trained weights
lemur = LemurNumPy()
lemur.load_npz("model.npz")

# Query embeddings come from ColBERT (not shown - you need this separately)
query_embeddings = your_colbert_model.encode_query(query_text)  # shape: (N_tokens, 128)
query_counts = np.array([len(query_embeddings)], dtype=np.int32)

# LEMUR projection + retrieval
features = lemur.compute_features(query_embeddings, query_counts)
indices, scores = lemur.top_k(features, k=100)
```

### Weight Export (Requires PyTorch)

On a machine with PyTorch and the original LEMUR:

```python
python lemur/export_weights.py /path/to/lemur_index /path/to/output.npz
```

Or manually:

```python
import torch
import numpy as np

mlp = torch.load("lemur_index/mlp.pt", map_location="cpu")
w = torch.load("lemur_index/w.pt", map_location="cpu")

np.savez_compressed("model.npz",
    layer_0_weight=mlp['state_dict']['feature_extractor.0.weight'].numpy(),
    layer_0_bias=mlp['state_dict']['feature_extractor.0.bias'].numpy(),
    layer_0_ln_weight=mlp['state_dict']['feature_extractor.1.weight'].numpy(),
    layer_0_ln_bias=mlp['state_dict']['feature_extractor.1.bias'].numpy(),
    W=w['W'].numpy(),
    final_hidden_dim=mlp['config']['final_hidden_dim']
)
```

## Realistic Use Case: When This Actually Helps

**Scenario:** You have an embedding service (API or microservice) that provides ColBERT embeddings, and you want the retrieval logic to be lightweight and dependency-minimal.

```
┌─────────────────────────────────┐
│  Embedding Service              │
│  (has GPU, runs ColBERT)        │
│  - Accepts text                 │
│  - Returns multi-vector embeds  │
└───────────────┬─────────────────┘
                │ embeddings via API
                ▼
┌─────────────────────────────────┐
│  Your Application (CPU only)    │
│  - LEMUR-NumPy (5MB)            │
│  - Projects to single vector    │
│  - pgvector/HNSW lookup         │
│  - Returns results              │
└─────────────────────────────────┘
```

In this architecture, LEMUR-NumPy lets you keep the retrieval layer lightweight while the heavy ColBERT inference happens elsewhere.

**If you don't have a separate embedding service**, you need ColBERT in your application anyway, at which point PyTorch is already a dependency and you might as well use the original LEMUR.

## Comparison: LEMUR vs Just Using Single-Vector

The paper shows LEMUR achieves ~95% recall relative to exact ColBERT MaxSim. But it doesn't show whether ColBERT → LEMUR beats modern single-vector models.

| Approach | Embedding | Retrieval | Complexity |
|----------|-----------|-----------|------------|
| OpenAI + pgvector | API call | HNSW | Simple |
| BGE + pgvector | 335M model | HNSW | Moderate |
| ColBERT + LEMUR + pgvector | 110M model | MLP + HNSW | Complex |

The ColBERT pipeline is only worth the complexity if you've validated it outperforms single-vector on your specific corpus. On BEIR benchmarks, the gap has narrowed to 1-3 nDCG points—LEMUR's approximation loss may consume that margin entirely.

**Recommendation:** Benchmark on your data before committing.

## Requirements

- Python 3.10+
- NumPy 1.20+
- Numba (strongly recommended for 3-4x speedup)

## References

- [LEMUR Paper](https://arxiv.org/abs/2601.21853) - Jääsaari et al., 2026
- [Original LEMUR Code](https://github.com/ejaasaari/lemur)
- [ColBERTv2](https://arxiv.org/abs/2112.01488) - Santhanam et al., 2022
