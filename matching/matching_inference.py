import os
import sys
from typing import Optional

import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
	sys.path.insert(0, _PROJECT_ROOT)

from datasets.no_split_dataloader import get_no_split_dataloader
from matching.matcher import matcher
from matching.matching_performance import fcv_frr
from matching_metrics import hybrid_metric
from matching.fingernet_wrapper import FingerNetWrapper
from matching.fingernet import FingerNet
from model.gumnet import GumNet


class IdentityAlignment(nn.Module):
	"""
	Alignment stub that returns the first batch unchanged.
	"""
	def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
		return img1

def gumnet_init():
	model = GumNet()
	model.load_state_dict(torch.load('gumnet.pth', map_location='cpu'))
	return model

def fingernet_init():
	fingernet = FingerNet()
	fingernet.load_state_dict(torch.load('fingernet.pth', map_location='cpu'))
	fingernet.eval()
	return fingernet

def build_matcher(device: torch.device, use_mask: bool = False) -> matcher:
	alignment = IdentityAlignment().to(device)
	fingernet = fingernet_init()
	fingernet.to(device)
	extractor = FingerNetWrapper(fingernet, minutiae_threshold=0.5, max_candidates=500).to(device)
	model = matcher(alignment, extractor, hybrid_metric, fcv_frr, mask=use_mask)
	model.eval()
	return model


def run_inference(
	data_root: str,
	batch_size: int = 8,
	num_workers: int = 2,
	max_batches: Optional[int] = None,
) -> None:
	# Limit CPU threads to avoid stalls from oversubscription
	torch.set_num_threads(1)
	torch.set_num_interop_threads(1)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	print("Building matcher model...")
	model = build_matcher(device=device)
	
	print("Creating dataloader...")
	loader = get_no_split_dataloader(
		data_root=data_root,
		batch_size=batch_size,
		num_workers=num_workers,
	)

	print("Running inference...")
	all_scores = []
	with torch.inference_mode():
		for batch_idx, batch in enumerate(loader):
			if batch_idx>=1:
				break
			Sa = batch["Sa"].to(device)
			Sb = batch["Sb"].to(device)

			scores = model(Sa, Sb)
			if not isinstance(scores, torch.Tensor):
				scores = torch.tensor(scores, dtype=torch.float32)
			all_scores.append(scores.detach().cpu())

			if max_batches is not None and (batch_idx + 1) >= max_batches:
				break

	if all_scores:
		scores_tensor = torch.cat([
			s.view(-1) if isinstance(s, torch.Tensor) else torch.tensor([s], dtype=torch.float32)
			for s in all_scores
		], dim=0)
		print(f"Processed {scores_tensor.numel()} samples.")
		print(f"Performance: {scores_tensor.mean().item():.4f}")


if __name__ == "__main__":
	_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
	DATA_ROOT = os.path.abspath(
		os.path.join(_SCRIPT_DIR, "..", "data", "FCV", "FVC2004", "Dbs", "DB1_A")
	)

	run_inference(
		data_root=DATA_ROOT,
		batch_size=8,
		num_workers=0,
		max_batches=None,
	)
