import torch
from pathlib import Path
from matcher import matcher
from matching_metrics import hybrid_metric
from matching_performance import fcv_frr, correct_match

PROJECT_ROOT = Path(__file__).resolve().parents[1]

total_batches = 10*8
correct = 0

def _avg_minutiae_count(features):
	sum, all_sum = 0, 0
	for M in features['minutiae']:
		sum += 1
		for m in M:
			all_sum += 1
	return all_sum / sum


for idx in range(10*8):
	INTERMEDIATES_PATH = PROJECT_ROOT / f"intermediates_batch_{idx}.pt"

	data = matcher.load_intermediates(str(INTERMEDIATES_PATH))
	alignment1 = data["alignment1"]
	original_img = data["origninal_img"]
	features1 = data["extractor1"]
	features2 = data["extractor2"]

	scores = hybrid_metric(alignment1, original_img, features1, features2, alpha=0.5, beta=0.5, delta=0.0, mask=False)
	correct += correct_match(scores, idx)
	print(f"Batch/Image {idx}: Correct: {correct}")

	avg_minutiae_f1 = _avg_minutiae_count(features1)
	avg_minutiae_f2 = _avg_minutiae_count(features1)

	score_tensor = torch.diag(scores)
	score_min = float(score_tensor.min())
	score_max = float(score_tensor.max())
	score_var = float(score_tensor.var(unbiased=False))

#	print(f"Avg minutiae count (features1): {avg_minutiae_f1:.4f}")
#	print(f"Avg minutiae count (features2): {avg_minutiae_f2:.4f}")
#	print(f"Score matrix min: {score_min:.6f}")
#	print(f"Score matrix max: {score_max:.6f}")
#	print(f"Score matrix var: {score_var:.6f}")
#	print(score_tensor[0:10])

accuracy = correct / total_batches
print(f"Overall accuracy: {accuracy:.4f}")