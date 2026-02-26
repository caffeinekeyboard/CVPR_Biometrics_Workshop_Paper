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
from matching_metrics import hybrid_metric, minutiae_metric, matching, calculate_true_matches, average_minutiae_match_distance
from matching.fingernet_wrapper import FingerNetWrapper
from matching.fingernet import FingerNet
from model.gumnet import GumNet
from datasets.minutiae_eval_data import get_minutiae_eval_dataloader
from datasets.minutiae_orig_data import get_minutiae_orig_dataloader
from nbis_extractor import Nbis


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = PROJECT_ROOT / "model" / "gumnet_2d_best_noise_level_0_8x8_200.pth"
FN_WEIGHTS_PATH = PROJECT_ROOT / "matching" / "fingernet.pth"

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
	batch_size: int = 8*8,
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
	#alignment = gumnet_init(device)
	extractor = Nbis().to(device)
	
	print("Creating dataloaders...")
	loader1 = get_minutiae_eval_dataloader(
		data_root=data_root,
		batch_size=batch_size,
		num_workers=num_workers,
	)
	loader2 = get_minutiae_orig_dataloader(
		data_root=data_root,
		batch_size=batch_size,
		num_workers=num_workers,
	)

	print("Running inference...")
	all_distances = []
	with torch.inference_mode():
		for idx, (eval, orig) in enumerate(zip(loader1, loader2)):
			if max_batches is not None and idx >= max_batches:
				break
			Sb = eval["Sb"].to(device)
			orig_Sb = orig["Sb"].to(device)
			Sa = eval["Sa"].to(device)
			orig_Sa = orig["Sa"].to(device)

			#_, control_points = alignment(Sa, Sb)
			#Sb_aligned = warp_original(orig_Sb, control_points)
			Sb_aligned = orig_Sb
			Sa_features = extractor(orig_Sa)
			Sb_features = extractor(Sb_aligned)
			
			average_distance = average_minutiae_match_distance(Sa_features["minutiae"], Sb_features["minutiae"])

			all_distances.append(average_distance)

			print(f"Average Minutiae Match Distance: {average_distance}")
	print("Inference completed.")
	print("Calculating performance...")

	average_distance = sum(all_distances) / len(all_distances)
	print(f"Average Minutiae Match Distance: {average_distance:.4f}")



if __name__ == "__main__":
	_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
	DATA_ROOT = os.path.abspath(
		os.path.join(_SCRIPT_DIR, "..", "data", "FCV", "FVC2004", "Dbs", "DB1_B")
	)

	run_inference(
		data_root=DATA_ROOT,
		max_batches=None,
	)
