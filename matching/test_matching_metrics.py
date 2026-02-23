import torch
from matching_metrics import minutiae_metric, orientation_metric

# Create dummy minutiae data in batches
# Each minutiae tensor: (num_minutiae, 4) with [x, y, angle, reliability]

# Batch 1: 3 fingerprints with different numbers of minutiae
batch1_fp1 = torch.tensor([
    [10.0, 20.0, 0.5, 0.9],
    [15.0, 25.0, 1.0, 0.8],
    [20.0, 30.0, 1.5, 0.7],
], dtype=torch.float32)

batch1_fp2 = torch.tensor([
    [10.2, 20.1, 0.52, 0.85],  # Close to fp1[0]
    [15.1, 24.9, 1.02, 0.75],  # Close to fp1[1]
], dtype=torch.float32)

batch1_fp3 = torch.tensor([
    [50.0, 60.0, 2.0, 0.9],
    [55.0, 65.0, 2.5, 0.8],
], dtype=torch.float32)

batch1 = [batch1_fp1, batch1_fp2, batch1_fp3]

# Batch 2: 2 fingerprints
batch2_fp1 = torch.tensor([
    [10.0, 20.0, 0.5, 0.9],
    [15.0, 25.0, 1.0, 0.8],
], dtype=torch.float32)

batch2_fp2 = torch.tensor([
    [50.0, 60.0, 2.0, 0.9],
    [55.0, 65.0, 2.5, 0.8],
    [60.0, 70.0, 3.0, 0.7],
], dtype=torch.float32)

batch2 = [batch2_fp1, batch2_fp2]

# Test 1: batch1 vs batch1 (should have high scores on diagonal)
print("Test 1: Batch 1 vs Batch 1")
scores_1 = minutiae_metric(batch1, batch1)
print(f"Shape: {scores_1.shape}")
print(f"Scores:\n{scores_1}\n")

# Test 2: batch1 vs batch2 (cross-batch comparison)
print("Test 2: Batch 1 vs Batch 2")
scores_2 = minutiae_metric(batch1, batch2)
print(f"Shape: {scores_2.shape}")
print(f"Scores:\n{scores_2}\n")

# Test 3: batch2 vs batch1
print("Test 3: Batch 2 vs Batch 1")
scores_3 = minutiae_metric(batch2, batch1)
print(f"Shape: {scores_3.shape}")
print(f"Scores:\n{scores_3}\n")

# Test 4: Empty minutiae handling
print("Test 4: With empty minutiae")
batch_empty = [batch1_fp1, torch.empty((0, 4), dtype=torch.float32)]
scores_4 = minutiae_metric(batch_empty, batch1)
print(f"Shape: {scores_4.shape}")
print(f"Scores:\n{scores_4}\n")

print("All minutiae tests completed successfully!\n")

# ============ Orientation Metric Tests ============
print("=" * 60)
print("ORIENTATION METRIC TESTS")
print("=" * 60 + "\n")

# Create dummy orientation field data
H, W = 128, 128

# Batch 1: 3 orientation fields
# Create orientation fields with different patterns
ori_batch1_1 = torch.ones(H, W) * 0.5  # Uniform orientation ~28.6°
ori_batch1_2 = torch.ones(H, W) * 0.0  # Uniform orientation ~0°
ori_batch1_3 = torch.ones(H, W) * torch.pi / 4  # Uniform orientation ~45°

batch1_ori = [ori_batch1_1, ori_batch1_2, ori_batch1_3]

# Batch 2: 2 orientation fields
# ori_batch2_1: similar to batch1_1 but with slight noise
ori_batch2_1 = torch.ones(H, W) * 0.5 + torch.randn(H, W) * 0.05
ori_batch2_1 = torch.clamp(ori_batch2_1, -torch.pi, torch.pi)

# ori_batch2_2: very different from all batch1
ori_batch2_2 = torch.ones(H, W) * torch.pi / 2  # ~90°

batch2_ori = [ori_batch2_1, ori_batch2_2]

# Test 1: Basic orientation metric without masks
print("Test 1: Orientation metric without masks")
ori_scores_1 = orientation_metric(batch1_ori, batch1_ori)
print(f"Shape: {ori_scores_1.shape}")
print(f"Scores (3x3, should have high diagonal):\n{ori_scores_1}\n")

# Test 2: Cross-batch comparison
print("Test 2: Cross-batch comparison (3x2)")
ori_scores_2 = orientation_metric(batch1_ori, batch2_ori)
print(f"Shape: {ori_scores_2.shape}")
print(f"Scores:\n{ori_scores_2}\n")

# Test 3: With segmentation masks
print("Test 3: With segmentation masks")
# Create masks (1 = valid region, 0 = invalid)
mask1_1 = torch.ones(H, W)
mask1_1[:64, :] = 0  # Mask out top half
mask1_2 = torch.ones(H, W)
mask1_3 = torch.ones(H, W)

batch1_masks = [mask1_1, mask1_2, mask1_3]

mask2_1 = torch.ones(H, W)
mask2_1[:64, :] = 0  # Same mask pattern as mask1_1
mask2_2 = torch.ones(H, W)

batch2_masks = [mask2_1, mask2_2]

ori_scores_3 = orientation_metric(
    batch1_ori, batch2_ori,
    mask_list_1=batch1_masks,
    mask_list_2=batch2_masks
)
print(f"Shape: {ori_scores_3.shape}")
print(f"Scores with masks:\n{ori_scores_3}\n")

# Test 4: Different angle thresholds
print("Test 4: Different angle thresholds")
threshold_tight = torch.pi / 12  # 15°
threshold_loose = torch.pi / 2   # 90°

ori_scores_tight = orientation_metric(batch1_ori, batch2_ori, angle_threshold=threshold_tight)
ori_scores_loose = orientation_metric(batch1_ori, batch2_ori, angle_threshold=threshold_loose)

print(f"Tight threshold (15°):\n{ori_scores_tight}")
print(f"\nLoose threshold (90°):\n{ori_scores_loose}")
print("\n(Looser threshold should generally give higher scores)\n")

# Test 5: Different sized orientation fields
print("Test 5: Different sized orientation fields")
ori_small = torch.ones(64, 64) * 0.5
ori_large = torch.ones(128, 128) * 0.5

different_size = orientation_metric([ori_small], [ori_large])
print(f"Score comparing 64x64 and 128x128 (same values): {different_size.item():.4f}")
print("(Should be ~1.0 since orientations are identical)\n")

# Test 6: Very different orientations
print("Test 6: Very different orientations")
ori_diff_1 = torch.ones(H, W) * 0.0
ori_diff_2 = torch.ones(H, W) * torch.pi  # Opposite direction

diff_score = orientation_metric([ori_diff_1], [ori_diff_2])
print(f"Score for opposite orientations (0 vs π): {diff_score.item():.4f}")
print("(Should be low/close to 0)\n")

print("All orientation metric tests completed successfully!")
