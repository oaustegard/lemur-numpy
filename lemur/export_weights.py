"""
Export LEMUR weights from PyTorch to NumPy format.

Run this script in an environment with PyTorch installed to convert
trained LEMUR models for use with lemur_numpy.py.

Usage:
    python export_weights.py /path/to/lemur_index /path/to/output.npz
"""

import sys
from pathlib import Path


def export_lemur_weights(index_path: Path, output_path: Path) -> None:
    """Export LEMUR weights from PyTorch checkpoints to .npz format."""
    import torch
    import numpy as np
    
    mlp_path = index_path / "mlp.pt"
    w_path = index_path / "w.pt"
    
    if not mlp_path.exists():
        raise FileNotFoundError(f"MLP checkpoint not found: {mlp_path}")
    if not w_path.exists():
        raise FileNotFoundError(f"W checkpoint not found: {w_path}")
    
    # Load PyTorch checkpoints
    mlp_data = torch.load(mlp_path, map_location="cpu", weights_only=False)
    w_data = torch.load(w_path, map_location="cpu", weights_only=False)
    
    state = mlp_data['state_dict']
    config = mlp_data['config']
    
    # Build save dict
    save_dict = {
        'output_mean': mlp_data.get('output_mean', 0.0),
        'output_std': mlp_data.get('output_std', 1.0),
        'final_hidden_dim': config['final_hidden_dim'],
        'W': w_data['W'].numpy() if isinstance(w_data['W'], torch.Tensor) else w_data['W'],
    }
    
    # Extract MLP layers
    # Structure: feature_extractor.{0,3,6,...} = Linear, then LN, then activation
    layers = []
    i = 0
    while f'feature_extractor.{i}.weight' in state:
        layers.append({
            'weight': state[f'feature_extractor.{i}.weight'].numpy(),
            'bias': state[f'feature_extractor.{i}.bias'].numpy(),
            'ln_weight': state[f'feature_extractor.{i+1}.weight'].numpy(),
            'ln_bias': state[f'feature_extractor.{i+1}.bias'].numpy(),
        })
        i += 3
    
    save_dict['num_layers'] = len(layers)
    for idx, layer in enumerate(layers):
        save_dict[f'layer_{idx}_weight'] = layer['weight']
        save_dict[f'layer_{idx}_bias'] = layer['bias']
        save_dict[f'layer_{idx}_ln_weight'] = layer['ln_weight']
        save_dict[f'layer_{idx}_ln_bias'] = layer['ln_bias']
    
    # Save as compressed .npz
    np.savez_compressed(output_path, **save_dict)
    
    # Print summary
    W = save_dict['W']
    print(f"Exported LEMUR weights to {output_path}")
    print(f"  MLP layers: {len(layers)}")
    print(f"  Hidden dim: {config['final_hidden_dim']}")
    print(f"  Input dim: {layers[0]['weight'].shape[1]}")
    print(f"  Corpus size: {W.shape[0]} documents")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python export_weights.py <lemur_index_dir> <output.npz>")
        print()
        print("Example:")
        print("  python export_weights.py ./lemur_index ./model.npz")
        sys.exit(1)
    
    index_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    export_lemur_weights(index_path, output_path)
