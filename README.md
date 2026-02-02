# LEMUR-NumPy: CPU-Only Multi-Vector Retrieval

A PyTorch-free implementation of [LEMUR](https://arxiv.org/abs/2601.21853) (Learned Multi-Vector Retrieval) for GPU-poor environments.

## TL;DR

**Can LEMUR run without PyTorch on CPU?** Yes. And as of Feb 2026, PyTorch CPU is also installable in Claude containers.

| Metric | Original (PyTorch + C++) | NumPy + Numba |
|--------|--------------------------|---------------|
| QPS (10k docs) | ~6900 | ~660 |
| Performance | 100% | ~10% |
| Dependencies | PyTorch (190MB), C++ ext | NumPy, Numba (~5MB) |
| Install size | 190MB | 5MB |
| GPU required | No | No |
| Training | Yes | No (weights only) |

## When to Use Which Version

**Use Original LEMUR (PyTorch):**
- Training new models in the container
- Maximum inference speed needed
- 190MB dependency acceptable
- Full research/development workflow

**Use LEMUR-NumPy:**
- Skills deployment (minimize dependency size)
- Inference-only use cases
- Multiple agent instances (lower memory per instance)
- Batch processing where 10x slower is fine

## Key Findings

### 1. LEMUR is CPU-Designed
The paper's experiments ran on Intel Xeon Gold 6230 CPUs with AVX-512. No GPU needed for inference—the design explicitly targets CPU deployment.

### 2. PyTorch Now Available (Feb 2026)
`download.pytorch.org` was added to the allowed domains, enabling PyTorch CPU installation:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu --break-system-packages
```

This means the original LEMUR implementation is now fully functional in Claude containers. However, the 190MB download makes NumPy version more practical for skills deployment.

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

## Installation

### Option 1: Original LEMUR (PyTorch)
```bash
# Fetch repo
curl -sL https://api.github.com/repos/ejaasaari/lemur/tarball/main | tar xz
cd ejaasaari-lemur-*

# Install PyTorch CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu --break-system-packages

# Install LEMUR
pip install -e . --break-system-packages
```

### Option 2: LEMUR-NumPy
```bash
# Fetch repo
curl -sL https://api.github.com/repos/oaustegard/lemur-numpy/tarball/main | tar xz
cd oaustegard-lemur-numpy-*

# Install (no PyTorch needed)
pip install numba --break-system-packages
pip install -e . --break-system-packages
```

## Usage

### Original LEMUR (PyTorch)
```python
from lemur import Lemur

# Train a model
lemur = Lemur(index="lemur_index", device="cpu")
lemur.fit(
    train=train_embeddings,      # shape: (total_tokens, embed_dim)
    train_counts=train_counts,   # shape: (num_docs,)
    epochs=10,
    verbose=True
)

# Query
test_features = lemur.compute_features((test_embeddings, test_counts))
scores = test_features @ lemur.W.T
top_k_indices = torch.topk(scores, k=100, dim=1).indices
```

### LEMUR-NumPy
```python
from lemur.lemur_numpy import LemurNumPy

# Option 1: Load pre-trained weights (exported from PyTorch)
lemur = LemurNumPy()
lemur.load_npz("model.npz")

# Option 2: Synthetic model for testing
from lemur.lemur_numpy import create_synthetic_model
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

To use a trained LEMUR model with the NumPy version, export weights on a machine with PyTorch:

```python
# On machine with PyTorch (after training)
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

## Complete Workflow Examples

### Workflow 1: Research & Development (Original LEMUR)

**Use case:** Experimenting with LEMUR on a research corpus

```python
# 1. Get embeddings from your favorite model
# (ColBERT, sentence-transformers, OpenAI API, etc.)
import numpy as np

# Example: 1000 documents, avg 50 tokens each, 128-dim embeddings
doc_embeddings = load_embeddings_from_somewhere()  # shape: (50000, 128)
doc_counts = np.array([50] * 1000, dtype=np.int32)

# 2. Install and train LEMUR (one-time setup)
from lemur import Lemur

lemur = Lemur(index="arxiv_papers_index", device="cpu")
lemur.fit(
    train=doc_embeddings,
    train_counts=doc_counts,
    epochs=10,
    hidden_dim=2048,
    verbose=True
)
# This saves mlp.pt and w.pt to arxiv_papers_index/

# 3. Query the index
query_text = "What are recent advances in transformer architectures?"
query_embeddings = embed_text(query_text)  # shape: (32, 128)
query_counts = np.array([32], dtype=np.int32)

features = lemur.compute_features((query_embeddings, query_counts))
scores = features @ lemur.W.T
top_10 = torch.topk(scores, k=10, dim=1)

# 4. Get results
for idx in top_10.indices[0]:
    print(f"Document {idx}: {get_doc_title(idx)}")
```

**Performance:** ~6900 QPS, 190MB dependencies  
**Best for:** Active research, iterating on hyperparameters, maximum speed

---

### Workflow 2: Skills Deployment (LEMUR-NumPy)

**Use case:** RAG skill that needs local semantic search without API calls

#### Phase 1: Training (External Machine - Claude Code)
```python
# In Claude Code or laptop with PyTorch
from lemur import Lemur
import numpy as np

# 1. Embed your knowledge base via API
docs = load_documentation()
embeddings = []
counts = []

for doc in docs:
    emb = openai_embed(doc)  # or Claude API, etc.
    embeddings.append(emb)
    counts.append(len(emb))

doc_embeddings = np.vstack(embeddings).astype(np.float32)
doc_counts = np.array(counts, dtype=np.int32)

# 2. Train LEMUR
lemur = Lemur(index="kb_index")
lemur.fit(doc_embeddings, doc_counts, epochs=10)

# 3. Export to NumPy format (critical step!)
import torch
mlp = torch.load("kb_index/mlp.pt", map_location="cpu")
w = torch.load("kb_index/w.pt", map_location="cpu")

np.savez_compressed("kb_model.npz",
    layer_0_weight=mlp['state_dict']['feature_extractor.0.weight'].numpy(),
    layer_0_bias=mlp['state_dict']['feature_extractor.0.bias'].numpy(),
    layer_0_ln_weight=mlp['state_dict']['feature_extractor.1.weight'].numpy(),
    layer_0_ln_bias=mlp['state_dict']['feature_extractor.1.bias'].numpy(),
    W=w['W'].numpy(),
    final_hidden_dim=mlp['config']['final_hidden_dim']
)
# Result: kb_model.npz (~15MB for 5000 docs)
```

#### Phase 2: Deployment (Skills Environment)
```python
# In skill - just numpy+numba, no PyTorch!
from lemur.lemur_numpy import LemurNumPy
import numpy as np

class RAGSkill:
    def __init__(self):
        # Load once at boot
        self.lemur = LemurNumPy()
        self.lemur.load_npz("/mnt/skills/user/rag/kb_model.npz")
        self.doc_embeddings = np.load("/mnt/skills/user/rag/docs.npz")
        
    def search(self, query_text: str, k: int = 10):
        # Embed query via API (only network call)
        query_emb = claude_embed_api(query_text)
        query_counts = np.array([len(query_emb)], dtype=np.int32)
        
        # Local search (no API calls, fast)
        features = self.lemur.compute_features(query_emb, query_counts)
        indices, scores = self.lemur.top_k(features, k=k)
        
        # Optional: exact reranking
        exact = self.lemur.exact_maxsim(
            query_emb, query_counts,
            self.doc_embeddings['embeddings'],
            self.doc_embeddings['counts'],
            indices
        )
        
        return indices[0], exact[0]

# Use it
rag = RAGSkill()
doc_ids, scores = rag.search("How do I configure memory settings?")
```

**Performance:** ~660 QPS, 5MB dependencies  
**Best for:** Skills, multi-agent systems, edge deployment, privacy-sensitive use cases

---

### Workflow 3: Hybrid Approach

**Use case:** Development with PyTorch, deployment with NumPy

```python
# Development (Claude.ai with PyTorch)
from lemur import Lemur

# Rapid iteration with full PyTorch
lemur = Lemur(index="dev_index")
lemur.fit(train_data, train_counts, epochs=5)
results = evaluate_on_test_set(lemur)  # Fast evaluation

# When satisfied, export
export_to_numpy("dev_index", "production.npz")

# ---

# Production (Skills with NumPy)
from lemur.lemur_numpy import LemurNumPy

lemur = LemurNumPy()
lemur.load_npz("production.npz")
# Deployed across 100 agent instances, 5MB each
```

## Use Cases

### 1. Agent-Local RAG
Embed documents externally (via API), train LEMUR externally, deploy weights to agent container for local semantic search without API calls.

**Why LEMUR-NumPy:** Minimize per-agent overhead when deploying to many instances.

### 2. Privacy-Sensitive Search
Documents embedded once, then searched locally without external API calls.

**Why LEMUR-NumPy:** No network dependencies after deployment.

### 3. Batch Processing
Overnight indexing jobs that don't justify GPU rental.

**Why either:** Both run on CPU. Choose NumPy for lower resource usage across many jobs.

### 4. Research & Experimentation
Quick iteration on retrieval methods, hyperparameter tuning.

**Why Original:** Full PyTorch for training, 10x faster inference for evaluations.

## Limitations

### LEMUR-NumPy
- **Training requires PyTorch** - only inference is PyTorch-free
- **~10x slower than optimized C++** - acceptable for many use cases
- **Memory scales with corpus** - W matrix is (num_docs × hidden_dim)

### Original LEMUR
- **190MB dependency** - large for skills deployment
- **C++ compilation required** - adds complexity to setup

## Architecture Comparison

### Original LEMUR
```
┌─────────────────────────────────────┐
│  Single Environment (PyTorch)       │
│  - Embed documents                  │
│  - Train LEMUR                      │
│  - Query with max performance       │
│  Dependencies: 190MB                │
└─────────────────────────────────────┘
```

### LEMUR-NumPy
```
┌─────────────────────────────────────┐
│  External (with PyTorch)            │
│  - Embed documents via ColBERT/API  │
│  - Train LEMUR projection matrix    │
│  - Export weights to .npz (15MB)    │
└───────────────┬─────────────────────┘
                │ .npz weights
                ▼
┌─────────────────────────────────────┐
│  Container/Edge (NumPy only)        │
│  - Load weights (5MB deps)          │
│  - Fast inference via Numba JIT     │
│  - Serve queries locally            │
│  - Deploy to many instances         │
└─────────────────────────────────────┘
```

## Requirements

### Original LEMUR
- Python 3.10+
- PyTorch 2.8+ (190MB)
- NumPy 2.2+
- C++ compiler for extensions

### LEMUR-NumPy
- Python 3.10+
- NumPy 1.20+
- Numba (strongly recommended, 3-4x speedup, ~5MB)
- AVX-512 CPU (recommended but not required)

## References

- [LEMUR Paper](https://arxiv.org/abs/2601.21853) - Jääsaari et al., 2026
- [Original Code](https://github.com/ejaasaari/lemur)
- [LEMUR-NumPy](https://github.com/oaustegard/lemur-numpy)
- [ColBERTv2](https://arxiv.org/abs/2112.01488) - Santhanam et al., 2022
