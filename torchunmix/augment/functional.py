from ..colorspace import rgb_to_stains, stains_to_rgb

from typing import Tuple, Optional
import torch


Ranges = Tuple[Tuple[float, float], Tuple[float, float], Optional[Tuple[float, float]]]


def augment_stains(rgb: torch.Tensor,
                   mat_to_stain: torch.Tensor,
                   mat_to_rgb: torch.Tensor,
                   mat_augment: torch.Tensor,
                   threshold: float = 0.03) -> torch.Tensor:
    stains = rgb_to_stains(rgb, mat_to_stain).permute(2, 3, 0, 1)
    mask = torch.where(stains < threshold)
    mask_vals = stains[mask]
    stains += mat_augment
    stains[mask] = mask_vals
    return stains_to_rgb(stains.permute(2, 3, 0, 1), mat_to_rgb)


def augment_stains_random(rgb: torch.Tensor,
                          mat_to_stain: torch.Tensor,
                          mat_to_rgb: torch.Tensor,
                          ranges: Ranges,
                          threshold: float = 0.03) -> torch.Tensor:
    mat_augment = torch.stack([
        torch.empty(rgb.shape[0], device=rgb.device).uniform_(*x) for x in ranges
    ]).transpose(0, 1)
    return augment_stains(rgb, mat_to_stain, mat_to_rgb, mat_augment, threshold=threshold)
