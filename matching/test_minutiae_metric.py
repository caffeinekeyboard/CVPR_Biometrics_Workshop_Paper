import torch

from matching_metrics import minutiae_metric


def _make_minutiae(coords, angles, device="cpu"):
    coords = torch.tensor(coords, dtype=torch.float32, device=device)
    angles = torch.tensor(angles, dtype=torch.float32, device=device).unsqueeze(1)
    return torch.cat([coords, angles], dim=1)


def test_minutiae_metric_basic():
    device = "cpu"

    # Batch 1: two samples
    m1a = _make_minutiae([[10, 10], [20, 20]], [0.0, 0.5], device)
    m1b = _make_minutiae([[5, 5], [15, 15], [25, 25]], [0.1, 0.6, 1.0], device)

    # Batch 2: three samples
    m2a = _make_minutiae([[10.5, 9.8], [19.9, 20.2]], [0.05, 0.55], device)
    m2b = _make_minutiae([[100, 100]], [2.0], device)
    m2c = _make_minutiae([[5.2, 5.1], [14.8, 15.3], [26, 24.9]], [0.15, 0.65, 1.05], device)

    batch1 = [m1a, m1b]
    batch2 = [m2a, m2b, m2c]

    scores = minutiae_metric(
        batch1,
        batch2,
        distance_threshold=2.0,
        angle_threshold=0.2,
        match_ratio_threshold=0.0,
    )

    assert scores.shape == (2, 3)

    # Similar pairs should score higher than dissimilar ones
    assert scores[0, 0] > scores[0, 1]
    assert scores[1, 2] > scores[1, 1]
