import torch
import numpy as np
from typing import Dict

def minutiae_metrics(output1: Dict[str, torch.Tensor], output2: Dict[str, torch.Tensor], 
                                 distance_threshold: float = 20.0, 
                                 angle_threshold: float = np.pi / 6) -> Dict[str, float]:
    """
    Match two fingerprints based on their minutiae maps using a heuristic approach.
    
    Args:
        output1: Output dictionary from FingerNetWrapper for first fingerprint
                 Contains 'minutiae' key with list of tensors (N1, 4) - [x, y, angle, score]
        output2: Output dictionary from FingerNetWrapper for second fingerprint
                 Contains 'minutiae' key with list of tensors (N2, 4) - [x, y, angle, score]
        distance_threshold: Maximum Euclidean distance for two minutiae to be considered matching
        angle_threshold: Maximum angular difference (in radians) for two minutiae to be considered matching
    
    Returns:
        Dictionary containing:
            - 'similarity_score': Normalized similarity score between 0 and 1
            - 'num_matches': Number of matched minutiae pairs
            - 'num_minutiae1': Total minutiae in fingerprint 1
            - 'num_minutiae2': Total minutiae in fingerprint 2
    """
    # Extract minutiae from both outputs (take first batch element)
    minutiae1 = output1['minutiae'][0]  # Shape: (N1, 4) - [x, y, angle, score]
    minutiae2 = output2['minutiae'][0]  # Shape: (N2, 4) - [x, y, angle, score]
    
    n1 = minutiae1.shape[0]
    n2 = minutiae2.shape[0]
    
    # Handle edge cases
    if n1 == 0 or n2 == 0:
        return {
            'similarity_score': 0.0,
            'num_matches': 0,
            'num_minutiae1': n1,
            'num_minutiae2': n2
        }
    
    # Extract coordinates and angles
    coords1 = minutiae1[:, :2]  # (N1, 2)
    coords2 = minutiae2[:, :2]  # (N2, 2)
    angles1 = minutiae1[:, 2]   # (N1,)
    angles2 = minutiae2[:, 2]   # (N2,)
    
    # Compute pairwise Euclidean distance matrix between all minutiae
    # Shape: (N1, N2)
    dist_matrix = torch.cdist(coords1, coords2)
    
    # Compute pairwise angular distance matrix
    # Using broadcasting to compute all pairs
    angles1_expanded = angles1.unsqueeze(1)  # (N1, 1)
    angles2_expanded = angles2.unsqueeze(0)  # (1, N2)
    angle_diff = torch.abs(angles1_expanded - angles2_expanded)
    # Handle circular nature of angles (shortest angular distance)
    angle_matrix = torch.minimum(angle_diff, 2 * np.pi - angle_diff)
    
    # Create matching matrix: True where both distance and angle are within thresholds
    matching_matrix = (dist_matrix < distance_threshold) & (angle_matrix < angle_threshold)
    
    # Find matches using greedy matching (each minutiae in fp1 can match at most one in fp2)
    num_matches = 0
    matched_in_fp2 = torch.zeros(n2, dtype=torch.bool, device=minutiae2.device)
    
    for i in range(n1):
        # Find all potential matches for minutiae i in fingerprint 1
        potential_matches = matching_matrix[i] & ~matched_in_fp2
        
        if potential_matches.any():
            # Find the closest match among potential matches
            potential_indices = torch.where(potential_matches)[0]
            distances_to_potentials = dist_matrix[i, potential_indices]
            best_match_idx = potential_indices[torch.argmin(distances_to_potentials)]
            
            # Mark this match
            matched_in_fp2[best_match_idx] = True
            num_matches += 1
    
    # Calculate normalized similarity score
    # Use the average of the number of minutiae as normalization factor
    avg_minutiae = (n1 + n2) / 2.0
    similarity_score = num_matches / avg_minutiae if avg_minutiae > 0 else 0.0
    
    # Alternatively, could use min or max for normalization:
    # similarity_score = num_matches / min(n1, n2) if min(n1, n2) > 0 else 0.0
    
    return {
        'similarity_score': float(similarity_score),
        'num_matches': int(num_matches),
        'num_minutiae1': int(n1),
        'num_minutiae2': int(n2)
    }


def orientation_map_metrics(output1: Dict[str, torch.Tensor], output2: Dict[str, torch.Tensor],
                           angle_threshold: float = np.pi / 6) -> Dict[str, float]:
    """
    Compare two fingerprints based on their orientation field maps.
    
    This metric computes similarity between dense orientation fields by comparing
    angular differences at corresponding spatial locations. Only valid regions
    (where both segmentation masks are non-zero) are considered.
    
    Args:
        output1: Output dictionary from FingerNetWrapper for first fingerprint
                 Contains 'orientation_field' key with tensor (B, H, W) in radians
                 and 'segmentation_mask' key with tensor (B, H, W)
        output2: Output dictionary from FingerNetWrapper for second fingerprint
                 Contains 'orientation_field' and 'segmentation_mask' keys
        angle_threshold: Maximum angular difference (in radians) for orientations 
                        to be considered matching. Default: π/6 (30 degrees)
    
    Returns:
        Dictionary containing:
            - 'similarity_score': Normalized similarity score between 0 and 1
            - 'num_matching_pixels': Number of pixels with matching orientations
            - 'num_valid_pixels': Total number of valid (overlapping mask) pixels
            - 'overlap_ratio': Ratio of overlapping mask area to total mask area
    """
    # Extract orientation fields and masks (work with full batch)
    ori_field1 = output1['orientation_field']  # Shape: (B, H, W)
    ori_field2 = output2['orientation_field']  # Shape: (B, H, W)
    
    # Get segmentation masks (convert from byte to boolean)
    mask1 = output1['segmentation_mask'] > 0  # Shape: (B, H, W)
    mask2 = output2['segmentation_mask'] > 0  # Shape: (B, H, W)
    
    # Ensure shapes match
    if ori_field1.shape != ori_field2.shape:
        raise ValueError(f"Orientation field shapes don't match: {ori_field1.shape} vs {ori_field2.shape}")
    
    # Find overlapping valid region (both masks are non-zero)
    valid_mask = mask1 & mask2  # Shape: (B, H, W)
    num_valid_pixels = valid_mask.sum().item()
    
    # Handle edge case: no overlap
    if num_valid_pixels == 0:
        return {
            'similarity_score': 0.0,
            'num_matching_pixels': 0,
            'num_valid_pixels': 0,
            'overlap_ratio': 0.0
        }
    
    # Extract orientations at valid pixels
    ori1_valid = ori_field1[valid_mask]  # Shape: (num_valid_pixels,)
    ori2_valid = ori_field2[valid_mask]  # Shape: (num_valid_pixels,)
    
    # Compute angular difference (accounting for circular nature of angles)
    angle_diff = torch.abs(ori1_valid - ori2_valid)
    angle_diff = torch.minimum(angle_diff, 2 * np.pi - angle_diff)
    
    # Count matching pixels (where angular difference is below threshold)
    matching_pixels = (angle_diff < angle_threshold).sum().item()
    
    # Calculate similarity score (ratio of matching pixels to valid pixels)
    similarity_score = matching_pixels / num_valid_pixels
    
    # Calculate overlap ratio (overlapping mask area vs total mask area)
    total_mask_area = (mask1 | mask2).sum().item()
    overlap_ratio = num_valid_pixels / total_mask_area if total_mask_area > 0 else 0.0
    
    return {
        'similarity_score': float(similarity_score),
        'num_matching_pixels': int(matching_pixels),
        'num_valid_pixels': int(num_valid_pixels),
        'overlap_ratio': float(overlap_ratio)
    }


def hybrid_match(alignment1: torch.Tensor, alignment2: torch.Tensor, extractor1: torch.Tensor, extractor2: torch.Tensor) -> torch.Tensor:
        combined_score = None  # TODO: Implement hybrid matching

        return combined_score