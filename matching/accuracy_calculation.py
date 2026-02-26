import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
	sys.path.insert(0, _PROJECT_ROOT)

from datasets.no_split_dataloader import get_no_split_dataloader
from matching.matcher import matcher, warp_original
from matching.matching_performance import fcv_frr
from matching_metrics import hybrid_metric, minutiae_metric, matching, calculate_true_matches
from matching.fingernet_wrapper import FingerNetWrapper
from matching.fingernet import FingerNet
from model.gumnet import GumNet
from datasets.eval_loader_data import EvalDataset, get_eval_dataloader
from datasets.original_data import OrigDataset, get_orig_dataloader
from datasets.accuracy_orig_data import AccDatasetOrig, get_acc_orig_dataloader
from datasets.accuracy_eval_data import AccDatasetEval, get_acc_eval_dataloader
from datasets.seq_orig_data import get_seq_orig_dataloader
from datasets.seq_eval_data import get_seq_eval_dataloader
from datasets.full_orig_data import get_full_orig_dataloader
from datasets.full_eval_data import get_full_eval_dataloader
from nbis_extractor import Nbis


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = PROJECT_ROOT / "model" / "gumnet_2d_best_noise_level_0_8x8_200.pth"
FN_WEIGHTS_PATH = PROJECT_ROOT / "matching" / "fingernet.pth"

class IdentityAlignment(nn.Module):
	"""
	Alignment stub that returns the first batch unchanged.
	"""
	def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
		return img1

def gumnet_init(device: torch.device):
	model = GumNet(grid_size=8)
	model.load_state_dict(WEIGHTS_PATH, map_location=device, strict=True)
	return model

def fingernet_init():
	fingernet = FingerNet()
	fingernet.load_state_dict(torch.load(FN_WEIGHTS_PATH, map_location='cpu'))
	return fingernet

def run_inference(
	data_root: str,
	batch_size: int = 10*8,
	num_workers: int = 0,
	max_batches: Optional[int] = None,
) -> None:
	# Limit CPU thread usage to reduce potential stalling in feature extraction
	try:
		torch.set_num_threads(1)
		torch.set_num_interop_threads(1)
	except Exception:
		pass
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	print("Building model...")
	alignment = gumnet_init(device)
	extractor = Nbis().to(device)
	
	print("Creating dataloaders...")
	template_loader_GN = get_seq_eval_dataloader(
		data_root=data_root,
		batch_size=1,
		num_workers=num_workers,
	)
	template_loader_orig = get_seq_orig_dataloader(
		data_root=data_root,
		batch_size=1,
		num_workers=num_workers,
	)
	impression_loader_GN =  get_full_eval_dataloader(
		data_root=data_root,
		batch_size=batch_size,
		num_workers=num_workers,
	)
	impression_loader_orig =  get_full_orig_dataloader(
		data_root=data_root,
		batch_size=batch_size,
		num_workers=num_workers,
	)

	print("Running inference...")
	all_true_matches = 0
	all_true_rejections = 0
	with torch.inference_mode():
		SB = iter(impression_loader_GN)
		ORIG_SB = iter(impression_loader_orig)
		impressions_gn = next(SB)
		impressions_orig = next(ORIG_SB)
		for template_idx, (template_gn, template_orig) in enumerate(zip(template_loader_GN, template_loader_orig)):
			if max_batches is not None and template_idx >= max_batches:
				break
			if template_idx == 10:
				break
			current_set = template_idx // 8
			Sb = impressions_gn["Sb"].to(device)
			orig_Sb = impressions_orig["Sb"].to(device)
			Sa = template_gn["Sa"].to(device)
			Sa = Sa.expand(Sb.shape[0], *Sa.shape[1:]).contiguous()
			orig_Sa = template_orig["Sa"].to(device)

			_, control_points = alignment(Sa, Sb)
			Sb_aligned = warp_original(orig_Sb, control_points)
			Sa_features = extractor(orig_Sa)
			Sb_features = extractor(Sb_aligned)

			score_tensor = matching(Sa_features, Sb_features, threshold=0.5)
			true_matches, true_rejections, _, _ = calculate_true_matches(current_set*8, (current_set+1)*8, score_tensor)
			all_true_matches += true_matches
			all_true_rejections += true_rejections

			print(f"True Matches: {true_matches}, True Rejections: {true_rejections}")

	print("Inference completed.")
	print("Calculating performance...")

	verification_accuracy = (all_true_matches + all_true_rejections) / (10*8*10*8)
	print(f"Verification Accuracy: {verification_accuracy:.4f}")



if __name__ == "__main__":
	_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
	DATA_ROOT = os.path.abspath(
		os.path.join(_SCRIPT_DIR, "..", "data", "FCV", "FVC2004", "Dbs", "DB1_B")
	)

	run_inference(
		data_root=DATA_ROOT,
		max_batches=None,
	)
