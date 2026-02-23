"""Test matching performance metrics."""

import torch
from matching_performance import (
    compute_far_frr,
    compute_far_frr_curve,
    find_eer_threshold,
    compute_rank_n_accuracy,
    compute_genuine_impostor_distributions
)


def test_compute_far_frr():
    """Test FAR/FRR computation."""
    # Create a simple 3x3 score matrix
    # Diagonal should be high (genuine matches), off-diagonal low (impostors)
    scores = torch.tensor([
        [0.9, 0.1, 0.2],
        [0.15, 0.85, 0.1],
        [0.05, 0.12, 0.95]
    ])
    
    far, frr = compute_far_frr(scores, threshold=0.5)
    print(f"FAR: {far:.4f}, FRR: {frr:.4f}")
    
    # At threshold 0.5, all genuine scores (0.9, 0.85, 0.95) are >= 0.5, so FRR = 0
    # All impostor scores are < 0.5, so FAR = 0
    assert frr == 0.0, f"Expected FRR=0.0, got {frr}"
    assert far == 0.0, f"Expected FAR=0.0, got {far}"
    print("✓ compute_far_frr passed")


def test_compute_far_frr_curve():
    """Test FAR/FRR curve generation."""
    scores = torch.rand(10, 10)
    # Make diagonal scores higher (genuine matches)
    torch.diagonal(scores).copy_(torch.rand(10) * 0.3 + 0.7)  # 0.7-1.0
    
    thresholds, far_vals, frr_vals = compute_far_frr_curve(scores, num_thresholds=100)
    
    print(f"Thresholds shape: {thresholds.shape}")
    print(f"FAR values shape: {far_vals.shape}")
    print(f"FRR values shape: {frr_vals.shape}")
    print(f"FAR range: [{far_vals.min():.4f}, {far_vals.max():.4f}]")
    print(f"FRR range: [{frr_vals.min():.4f}, {frr_vals.max():.4f}]")
    
    assert thresholds.shape[0] == 100
    assert far_vals.shape[0] == 100
    assert frr_vals.shape[0] == 100
    assert far_vals.min() >= 0.0 and far_vals.max() <= 1.0
    assert frr_vals.min() >= 0.0 and frr_vals.max() <= 1.0
    print("✓ compute_far_frr_curve passed")


def test_find_eer_threshold():
    """Test EER threshold finding."""
    scores = torch.rand(50, 50)
    # Make diagonal scores higher
    torch.diagonal(scores).copy_(torch.rand(50) * 0.3 + 0.7)
    
    eer, threshold, idx = find_eer_threshold(scores, num_thresholds=500)
    
    print(f"EER: {eer:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"EER index: {idx}")
    
    assert 0.0 <= eer <= 1.0
    assert 0.0 <= threshold <= 1.0
    assert 0 <= idx < 500
    print("✓ find_eer_threshold passed")


def test_compute_rank_n_accuracy():
    """Test rank-N accuracy."""
    # Perfect matching scenario
    scores = torch.eye(5) * 10.0 + torch.rand(5, 5) * 0.1
    
    rank1_acc = compute_rank_n_accuracy(scores, n=1)
    rank3_acc = compute_rank_n_accuracy(scores, n=3)
    
    print(f"Rank-1 accuracy: {rank1_acc:.4f}")
    print(f"Rank-3 accuracy: {rank3_acc:.4f}")
    
    # Should be perfect since diagonal is much higher
    assert rank1_acc == 1.0, f"Expected rank-1 accuracy=1.0, got {rank1_acc}"
    assert rank3_acc == 1.0
    print("✓ compute_rank_n_accuracy passed")


def test_compute_genuine_impostor_distributions():
    """Test score distribution extraction."""
    n = 20
    scores = torch.rand(n, n)
    torch.diagonal(scores).copy_(torch.rand(n) * 0.3 + 0.7)
    
    genuine, impostor = compute_genuine_impostor_distributions(scores)
    
    print(f"Genuine scores shape: {genuine.shape}")
    print(f"Impostor scores shape: {impostor.shape}")
    print(f"Genuine mean: {genuine.mean():.4f}, std: {genuine.std():.4f}")
    print(f"Impostor mean: {impostor.mean():.4f}, std: {impostor.std():.4f}")
    
    assert genuine.shape[0] == n
    assert impostor.shape[0] == n * (n - 1)
    assert genuine.mean() > impostor.mean()  # Genuine should be higher
    print("✓ compute_genuine_impostor_distributions passed")


def test_gpu_compatibility():
    """Test GPU support if available."""
    if not torch.cuda.is_available():
        print("⚠ GPU not available, skipping GPU test")
        return
    
    scores = torch.rand(100, 100).cuda()
    torch.diagonal(scores).copy_(torch.rand(100).cuda() * 0.3 + 0.7)
    
    # Test all functions on GPU
    far, frr = compute_far_frr(scores, threshold=0.5)
    thresholds, far_vals, frr_vals = compute_far_frr_curve(scores, num_thresholds=100)
    eer, threshold, idx = find_eer_threshold(scores)
    rank1_acc = compute_rank_n_accuracy(scores, n=1)
    genuine, impostor = compute_genuine_impostor_distributions(scores)
    
    print(f"GPU test - EER: {eer:.4f}, Rank-1: {rank1_acc:.4f}")
    assert thresholds.device.type == 'cuda'
    assert genuine.device.type == 'cuda'
    print("✓ GPU compatibility passed")


if __name__ == "__main__":
    print("Testing matching performance metrics...\n")
    
    test_compute_far_frr()
    print()
    
    test_compute_far_frr_curve()
    print()
    
    test_find_eer_threshold()
    print()
    
    test_compute_rank_n_accuracy()
    print()
    
    test_compute_genuine_impostor_distributions()
    print()
    
    test_gpu_compatibility()
    print()
    
    print("=" * 50)
    print("All tests passed! ✓")
