"""Utilities for saving visualizations."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
import torchvision


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ensure_batch(tensor: torch.Tensor) -> torch.Tensor:
	if not isinstance(tensor, torch.Tensor):
		raise TypeError("input must be a torch.Tensor")
	if tensor.dim() == 4:
		return tensor
	if tensor.dim() == 3:
		return tensor.unsqueeze(0)
	if tensor.dim() == 2:
		return tensor.unsqueeze(0).unsqueeze(0)
	raise ValueError("input tensor must have 2, 3, or 4 dimensions")


def send_to_visualize(
	batch_a: torch.Tensor,
	batch_b: torch.Tensor,
	batch_id: Union[int, str],
	output_dir: Union[str, Path, None] = None,
) -> Path:
	"""
	Concatenate the first image from each batch and save it to disk.

	Args:
		batch_a: Tensor with shape (B, C, H, W), (C, H, W), or (H, W).
		batch_b: Tensor with shape (B, C, H, W), (C, H, W), or (H, W).
		batch_id: Identifier used in the output filename.
		output_dir: Directory to save the image (defaults to <project>/visualizations).

	Returns:
		Path to the saved image.
	"""
	batch_a = _ensure_batch(batch_a)
	batch_b = _ensure_batch(batch_b)

	image_a = batch_a[0]
	image_b = batch_b[0]

	if image_a.dim() == 2:
		image_a = image_a.unsqueeze(0)
	if image_b.dim() == 2:
		image_b = image_b.unsqueeze(0)

	if image_a.dim() != 3 or image_b.dim() != 3:
		raise ValueError("images must have shape (C, H, W) after selection")

	if image_a.shape[0] != image_b.shape[0]:
		raise ValueError("images must have the same number of channels")

	if image_a.shape[1:] != image_b.shape[1:]:
		image_b = F.interpolate(
			image_b.unsqueeze(0),
			size=image_a.shape[1:],
			mode="bilinear" if image_b.shape[0] > 1 else "bicubic",
			align_corners=False,
		).squeeze(0)

	side_by_side = torch.cat([image_a, image_b], dim=-1).detach().cpu()

	save_dir = Path(output_dir) if output_dir is not None else (_PROJECT_ROOT / "visualizations")
	save_dir.mkdir(parents=True, exist_ok=True)
	save_path = save_dir / f"batch_{batch_id}.png"

	torchvision.utils.save_image(side_by_side, save_path)
	return save_path


__all__ = ["send_to_visualize"]
