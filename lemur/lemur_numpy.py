"""
LEMUR-NumPy: CPU-only Multi-Vector Retrieval

A PyTorch-free implementation of LEMUR (Learned Multi-Vector Retrieval)
for GPU-poor environments. Uses Numba JIT compilation for performance.

Performance: ~12% of original PyTorch+C++ (830 vs 6924 QPS on ArguAna-scale)
Requirements: numpy, numba

Usage:
    from lemur_numpy import LemurNumPy
    
    # Load pre-exported weights
    lemur = LemurNumPy()
    lemur.load_npz("model.npz")
    
    # Query
    query_features = lemur.compute_features(query_embeddings, query_counts)
    indices, scores = lemur.top_k(query_features, k=100)
    
    # Rerank with exact MaxSim
    exact_scores = lemur.exact_maxsim(
        query_embeddings, query_counts,
        doc_embeddings, doc_counts,
        indices
    )
"""

from __future__ import annotations
from typing import Optional, Tuple
from pathlib import Path
import numpy as np

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Warning: Numba not available. Performance will be significantly degraded.")


# ============================================================================
# Numba-optimized kernels
# ============================================================================

if HAS_NUMBA:
    @njit(parallel=True, fastmath=True)
    def _forward_pass_numba(
        query_embeddings: np.ndarray,
        W_proj: np.ndarray,
        b_proj: np.ndarray,
        ln_w: np.ndarray,
        ln_b: np.ndarray,
        offsets: np.ndarray,
        num_queries: int,
        hidden_dim: int
    ) -> np.ndarray:
        """Fused forward pass: Linear -> LayerNorm -> GELU -> Pool"""
        total_tokens = query_embeddings.shape[0]
        embed_dim = query_embeddings.shape[1]
        
        features = np.empty((total_tokens, hidden_dim), dtype=np.float32)
        
        for i in prange(total_tokens):
            # Linear
            for j in range(hidden_dim):
                acc = b_proj[j]
                for k in range(embed_dim):
                    acc += query_embeddings[i, k] * W_proj[j, k]
                features[i, j] = acc
            
            # LayerNorm
            mean = 0.0
            for j in range(hidden_dim):
                mean += features[i, j]
            mean /= hidden_dim
            
            var = 0.0
            for j in range(hidden_dim):
                diff = features[i, j] - mean
                var += diff * diff
            var /= hidden_dim
            
            inv_std = 1.0 / np.sqrt(var + 1e-5)
            for j in range(hidden_dim):
                features[i, j] = (features[i, j] - mean) * inv_std * ln_w[j] + ln_b[j]
            
            # GELU
            for j in range(hidden_dim):
                val = features[i, j]
                t = 0.7978845608028654 * (val + 0.044715 * val * val * val)
                features[i, j] = 0.5 * val * (1.0 + np.tanh(t))
        
        # Pool by query
        query_features = np.zeros((num_queries, hidden_dim), dtype=np.float32)
        for i in prange(num_queries):
            start = offsets[i]
            end = offsets[i + 1]
            for j in range(start, end):
                for k in range(hidden_dim):
                    query_features[i, k] += features[j, k]
            for k in range(hidden_dim):
                query_features[i, k] *= 0.03125  # 1/32
        
        return query_features

    @njit(parallel=True, fastmath=True)
    def _exact_maxsim_numba(
        query_embeddings: np.ndarray,
        q_offsets: np.ndarray,
        doc_embeddings: np.ndarray,
        d_offsets: np.ndarray,
        candidate_indices: np.ndarray
    ) -> np.ndarray:
        """Exact MaxSim computation for reranking candidates."""
        num_queries = candidate_indices.shape[0]
        num_candidates = candidate_indices.shape[1]
        embed_dim = query_embeddings.shape[1]
        
        scores = np.zeros((num_queries, num_candidates), dtype=np.float32)
        
        for q_idx in prange(num_queries):
            q_start = q_offsets[q_idx]
            q_end = q_offsets[q_idx + 1]
            num_q_tokens = q_end - q_start
            
            for c_idx in range(num_candidates):
                doc_idx = candidate_indices[q_idx, c_idx]
                d_start = d_offsets[doc_idx]
                d_end = d_offsets[doc_idx + 1]
                
                maxsim = 0.0
                for qi in range(num_q_tokens):
                    q_token = q_start + qi
                    max_sim_for_token = -1e9
                    
                    for di in range(d_start, d_end):
                        sim = 0.0
                        for k in range(embed_dim):
                            sim += query_embeddings[q_token, k] * doc_embeddings[di, k]
                        if sim > max_sim_for_token:
                            max_sim_for_token = sim
                    
                    maxsim += max_sim_for_token
                
                scores[q_idx, c_idx] = maxsim
        
        return scores


# ============================================================================
# Pure NumPy fallbacks
# ============================================================================

def _gelu_numpy(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))


def _layer_norm_numpy(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return weight * ((x - mean) / np.sqrt(var + 1e-5)) + bias


def _forward_pass_numpy(
    query_embeddings: np.ndarray,
    W_proj: np.ndarray,
    b_proj: np.ndarray,
    ln_w: np.ndarray,
    ln_b: np.ndarray,
    offsets: np.ndarray,
    num_queries: int,
    hidden_dim: int
) -> np.ndarray:
    """Pure NumPy forward pass (slow fallback)."""
    x = query_embeddings @ W_proj.T + b_proj
    x = _layer_norm_numpy(x, ln_w, ln_b)
    x = _gelu_numpy(x)
    
    query_features = np.zeros((num_queries, hidden_dim), dtype=np.float32)
    for i in range(num_queries):
        query_features[i] = x[offsets[i]:offsets[i+1]].sum(axis=0) / 32.0
    
    return query_features


# ============================================================================
# Main class
# ============================================================================

class LemurNumPy:
    """NumPy-only LEMUR for CPU inference."""
    
    def __init__(self):
        self.W_proj: Optional[np.ndarray] = None
        self.b_proj: Optional[np.ndarray] = None
        self.ln_w: Optional[np.ndarray] = None
        self.ln_b: Optional[np.ndarray] = None
        self.W_out: Optional[np.ndarray] = None
        self.hidden_dim: int = 0
        self.embed_dim: int = 0
        self._compiled: bool = False
    
    def load_npz(self, path: Path) -> "LemurNumPy":
        """Load model from .npz file."""
        data = np.load(path, allow_pickle=True)
        
        # Load single-layer MLP (LEMUR uses 1 hidden layer)
        self.W_proj = np.ascontiguousarray(data['layer_0_weight'], dtype=np.float32)
        self.b_proj = np.ascontiguousarray(data['layer_0_bias'], dtype=np.float32)
        self.ln_w = np.ascontiguousarray(data['layer_0_ln_weight'], dtype=np.float32)
        self.ln_b = np.ascontiguousarray(data['layer_0_ln_bias'], dtype=np.float32)
        self.W_out = np.ascontiguousarray(data['W'], dtype=np.float32)
        
        self.hidden_dim = int(data['final_hidden_dim'])
        self.embed_dim = self.W_proj.shape[1]
        
        return self
    
    def _ensure_compiled(self) -> None:
        """Trigger Numba JIT compilation if not done."""
        if not HAS_NUMBA or self._compiled:
            return
        
        # Warmup compilation
        dummy = np.random.randn(10, self.embed_dim).astype(np.float32)
        _ = _forward_pass_numba(
            dummy, self.W_proj, self.b_proj, self.ln_w, self.ln_b,
            np.array([0, 5, 10], dtype=np.int64), 2, self.hidden_dim
        )
        self._compiled = True
    
    def compute_features(
        self,
        embeddings: np.ndarray,
        counts: np.ndarray
    ) -> np.ndarray:
        """Compute pooled features for queries.
        
        Args:
            embeddings: Token embeddings, shape (total_tokens, embed_dim)
            counts: Tokens per query, shape (num_queries,)
            
        Returns:
            Query features, shape (num_queries, hidden_dim)
        """
        self._ensure_compiled()
        
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        num_queries = len(counts)
        
        if HAS_NUMBA:
            return _forward_pass_numba(
                embeddings, self.W_proj, self.b_proj, self.ln_w, self.ln_b,
                offsets, num_queries, self.hidden_dim
            )
        else:
            return _forward_pass_numpy(
                embeddings, self.W_proj, self.b_proj, self.ln_w, self.ln_b,
                offsets, num_queries, self.hidden_dim
            )
    
    def approximate_scores(self, query_features: np.ndarray) -> np.ndarray:
        """Compute approximate MaxSim scores for all documents.
        
        Returns:
            Scores, shape (num_queries, num_docs)
        """
        return query_features @ self.W_out.T
    
    def top_k(
        self,
        query_features: np.ndarray,
        k: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get top-k candidate documents.
        
        Returns:
            (indices, scores) - both shape (num_queries, k)
        """
        scores = self.approximate_scores(query_features)
        
        if k >= scores.shape[1]:
            indices = np.argsort(-scores, axis=1)
            sorted_scores = np.take_along_axis(scores, indices, axis=1)
            return indices, sorted_scores
        
        # O(n) partial sort
        indices = np.argpartition(-scores, k, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, indices, axis=1)
        
        # Sort within top-k
        order = np.argsort(-top_scores, axis=1)
        sorted_indices = np.take_along_axis(indices, order, axis=1)
        sorted_scores = np.take_along_axis(top_scores, order, axis=1)
        
        return sorted_indices, sorted_scores
    
    def exact_maxsim(
        self,
        query_embeddings: np.ndarray,
        query_counts: np.ndarray,
        doc_embeddings: np.ndarray,
        doc_counts: np.ndarray,
        candidate_indices: np.ndarray
    ) -> np.ndarray:
        """Compute exact MaxSim for reranking candidates.
        
        Args:
            query_embeddings: Query token embeddings
            query_counts: Tokens per query
            doc_embeddings: Document token embeddings
            doc_counts: Tokens per document
            candidate_indices: Shape (num_queries, k) - candidates to rerank
            
        Returns:
            Exact MaxSim scores, shape (num_queries, k)
        """
        query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        doc_embeddings = np.ascontiguousarray(doc_embeddings, dtype=np.float32)
        
        q_offsets = np.concatenate([[0], np.cumsum(query_counts)]).astype(np.int64)
        d_offsets = np.concatenate([[0], np.cumsum(doc_counts)]).astype(np.int64)
        
        if HAS_NUMBA:
            return _exact_maxsim_numba(
                query_embeddings, q_offsets,
                doc_embeddings, d_offsets,
                candidate_indices.astype(np.int64)
            )
        else:
            # Slow fallback
            num_queries, num_candidates = candidate_indices.shape
            scores = np.zeros((num_queries, num_candidates), dtype=np.float32)
            
            for q_idx in range(num_queries):
                q_tokens = query_embeddings[q_offsets[q_idx]:q_offsets[q_idx+1]]
                for c_idx, doc_idx in enumerate(candidate_indices[q_idx]):
                    d_tokens = doc_embeddings[d_offsets[doc_idx]:d_offsets[doc_idx+1]]
                    sim = q_tokens @ d_tokens.T
                    scores[q_idx, c_idx] = sim.max(axis=1).sum()
            
            return scores
    
    @property
    def num_docs(self) -> int:
        """Number of indexed documents."""
        return self.W_out.shape[0] if self.W_out is not None else 0


# ============================================================================
# Utility: Create synthetic model for testing
# ============================================================================

def create_synthetic_model(
    embed_dim: int = 128,
    hidden_dim: int = 2048,
    num_docs: int = 10000,
    seed: int = 42
) -> LemurNumPy:
    """Create a synthetic LEMUR model for testing."""
    np.random.seed(seed)
    
    lemur = LemurNumPy()
    lemur.embed_dim = embed_dim
    lemur.hidden_dim = hidden_dim
    lemur.W_proj = np.ascontiguousarray(
        np.random.randn(hidden_dim, embed_dim).astype(np.float32) * 0.02
    )
    lemur.b_proj = np.zeros(hidden_dim, dtype=np.float32)
    lemur.ln_w = np.ones(hidden_dim, dtype=np.float32)
    lemur.ln_b = np.zeros(hidden_dim, dtype=np.float32)
    lemur.W_out = np.ascontiguousarray(
        np.random.randn(num_docs, hidden_dim).astype(np.float32) * 0.02
    )
    
    return lemur


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    import time
    
    print("LEMUR-NumPy Test")
    print("=" * 50)
    print(f"Numba available: {HAS_NUMBA}")
    
    # Create model
    lemur = create_synthetic_model(
        embed_dim=128,
        hidden_dim=2048,
        num_docs=10000
    )
    
    # Create test data
    num_queries = 100
    tokens_per_query = 32
    query_embeddings = np.random.randn(
        num_queries * tokens_per_query, 128
    ).astype(np.float32)
    query_counts = np.full(num_queries, tokens_per_query, dtype=np.int32)
    
    # Warmup
    print("\nWarming up JIT...")
    _ = lemur.compute_features(query_embeddings[:32], np.array([32], dtype=np.int32))
    
    # Benchmark
    print("\nBenchmarking...")
    k = 100
    iterations = 10
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        features = lemur.compute_features(query_embeddings, query_counts)
        indices, scores = lemur.top_k(features, k=k)
    
    avg_time = (time.perf_counter() - t0) / iterations
    qps = num_queries / avg_time
    
    print(f"\nResults (10k docs, 100 queries, k={k}):")
    print(f"  Time: {avg_time*1000:.1f}ms")
    print(f"  QPS: {qps:.0f}")
    print(f"  vs Paper (6924 QPS): {qps/6924*100:.1f}%")
