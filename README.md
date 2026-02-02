# LEMUR-NumPy: CPU-Only Multi-Vector Retrieval

A PyTorch-free implementation of [LEMUR](https://arxiv.org/abs/2601.21853) (Learned Multi-Vector Retrieval) for GPU-poor environments.

## TL;DR

**Can LEMUR run without PyTorch on CPU?** Yes, with caveats.

| Metric | Original (PyTorch + C++) | NumPy + Numba |
|--------|--------------------------|---------------|
| QPS (10k docs) | ~6900 | ~660 |
| Performance | 100% | ~10% |
| Dependencies | PyTorch, AVX-512 C++ ext | NumPy, Numba |
| GPU required | No | No |

## Key Findings

### 1. LEMUR is CPU-Designed
The paper's experiments ran on Intel Xeon Gold 6230 CPUs with AVX-512. No GPU needed for inference—the design explicitly targets CPU deployment.

### 2. The PyTorch Blocker is Real
In restricted container environments (like Claude's), PyTorch installation fails due to network policy blocking `download.pytorch.org`. This motivated the NumPy-only approach.

### 3. Performance Gap Analysis

The 10x performance gap comes from:
- **PyTorch compiled kernels** vs interpreted NumPy
- **Custom C++ MaxSim extension** with AVX-512 intrinsics
- **Memory management** optimized in PyTorch

Numba JIT compilation recovers ~3-4x vs pure NumPy, but can't match hand-tuned C++.

### 4. This Container Has AVX-512
```
avx512f avx512dq avx512cd avx512bw avx512vl avx512vbmi avx512_vbmi2 
avx512_vnni avx512_bitalg avx512_vpopcntdq
```
The CPU supports all the vector instructions LEMUR needs. The bottleneck is software, not hardware.

## Usage

```python
from lemur_numpy import LemurNumPy, create_synthetic_model

# Option 1: Load pre-trained weights (exported from PyTorch)
lemur = LemurNumPy()
lemur.load_npz("model.npz")

# Option 2: Synthetic model for testing
lemur = create_synthetic_model(embed_dim=128, hidden_dim=2048, num_docs=10000)

# Query
query_embeddings = np.random.randn(100, 128).astype(np.float32)  # 100 tokens
query_counts = np.array([100], dtype=np.int32)  # 1 query with 100 tokens

# Get approximate candidates
features = lemur.compute_features(query_embeddings, query_counts)
indices, scores = lemur.top_k(features, k=100)

# Rerank with exact MaxSim (optional)
exact_scores = lemur.exact_maxsim(
    query_embeddings, query_counts,
    doc_embeddings, doc_counts,
    indices
)
```

## Weight Export

To use a trained LEMUR model, export weights on a machine with PyTorch:

```python
# On machine with PyTorch
import torch
import numpy as np

# Load trained LEMUR
payload = torch.load("lemur_index/mlp.pt", map_location="cpu")
w_payload = torch.load("lemur_index/w.pt", map_location="cpu")

# Export to NumPy format
save_dict = {
    'layer_0_weight': payload['state_dict']['feature_extractor.0.weight'].numpy(),
    'layer_0_bias': payload['state_dict']['feature_extractor.0.bias'].numpy(),
    'layer_0_ln_weight': payload['state_dict']['feature_extractor.1.weight'].numpy(),
    'layer_0_ln_bias': payload['state_dict']['feature_extractor.1.bias'].numpy(),
    'W': w_payload['W'].numpy(),
    'final_hidden_dim': payload['config']['final_hidden_dim'],
}
np.savez_compressed("model.npz", **save_dict)
```

## Use Cases

### 1. Agent-Local RAG
Embed documents externally (via API), train LEMUR externally, deploy weights to agent container for local semantic search without API calls.

### 2. Edge Deployment
Privacy-sensitive environments where GPU provisioning is expensive/impossible.

### 3. Batch Processing
Overnight indexing jobs that don't justify GPU rental.

## Limitations

- **Training requires PyTorch** - only inference is PyTorch-free
- **~10x slower than optimized C++** - acceptable for many use cases
- **Memory scales with corpus** - W matrix is (num_docs × hidden_dim)

## Architecture

```
┌─────────────────────────────────────┐
│  External (with GPU/PyTorch)        │
│  - Embed documents via ColBERT      │
│  - Train LEMUR projection matrix    │
│  - Export weights to .npz           │
└───────────────┬─────────────────────┘
                │ .npz weights
                ▼
┌─────────────────────────────────────┐
│  Container/Edge (NumPy only)        │
│  - Load weights                     │
│  - Fast inference via Numba JIT     │
│  - Serve queries locally            │
└─────────────────────────────────────┘
```

## Requirements

- Python 3.10+
- NumPy 1.20+
- Numba (strongly recommended, 3-4x speedup)
- AVX-512 CPU (recommended but not required)

## References

- [LEMUR Paper](https://arxiv.org/abs/2601.21853) - Jääsaari et al., 2026
- [Original Code](https://github.com/ejaasaari/lemur)
- [ColBERTv2](https://arxiv.org/abs/2112.01488) - Santhanam et al., 2022
