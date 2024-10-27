import math
import random
from typing import Tuple

import torch
import torch.nn.functional as F


def stochastic_patch_replacement(
    tensor: torch.Tensor,
    patch_size: Tuple[int, int],
    eps: float = 0.05
) -> torch.Tensor:
    """
    Perform Stochastic Patch Replacement (SPR) on a 4D tensor by randomly replacing a
    specified percentage of patches with normalized patches based on different statistical measures.

    Args:
        tensor (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        patch_size (Tuple[int, int]): Size of each patch as (patch_height, patch_width).
        eps (float, optional): Fraction of patches to replace (default is 0.05).

    Returns:
        torch.Tensor: Tensor after applying stochastic patch replacement with the same shape as input.

    Raises:
        ValueError: If input tensor does not have 4 dimensions.
        ValueError: If patch size is incompatible with tensor dimensions.
        ValueError: If eps is not in the range (0, 1).
    """
    if tensor.dim() != 4:
        raise ValueError(f"Input tensor must be 4D, but got {tensor.dim()}D.")

    if not (0.0 < eps < 1.0):
        raise ValueError(f"eps must be between 0 and 1, but got {eps}.")

    batch_size, channels, height, width = tensor.shape
    patch_height, patch_width = patch_size

    if height % patch_height != 0 or width % patch_width != 0:
        raise ValueError(
            "Patch size must evenly divide the tensor's height and width."
        )

    # Calculate the number of patches along height and width
    num_patches_h = height // patch_height
    num_patches_w = width // patch_width
    total_patches = num_patches_h * num_patches_w

    # Calculate number of patches to replace per image in the batch
    num_replace = max(1, math.ceil(total_patches * eps))

    # Unfold the tensor into patches
    patches = F.unfold(
        tensor,
        kernel_size=patch_size,
        stride=patch_size
    )  # Shape: (batch_size, channels * patch_height * patch_width, num_patches)

    patches = patches.view(
        batch_size, channels, patch_height, patch_width, total_patches
    )  # Shape: (batch_size, channels, patch_h, patch_w, num_patches)

    for b in range(batch_size):
        # Randomly select patches to replace
        replace_indices = random.sample(range(total_patches), num_replace)

        for idx in replace_indices:
            patch = patches[b, :, :, :, idx]

            # Choose a normalization method randomly
            norm_method = random.choice(['mean', 'median', 'variance'])

            if norm_method == 'mean':
                mean = patch.mean(dim=(-2, -1), keepdim=True)
                normalized_patch = patch - mean
            elif norm_method == 'median':
                median = patch.median(dim=-1)[0].median(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
                normalized_patch = patch - median
            elif norm_method == 'variance':
                var = patch.var(dim=(-2, -1), keepdim=True) + 1e-5  # Prevent division by zero
                normalized_patch = patch / var.sqrt()
            else:
                # Fallback to mean normalization
                mean = patch.mean(dim=(-2, -1), keepdim=True)
                normalized_patch = patch - mean

            patches[b, :, :, :, idx] = normalized_patch

    # Reshape patches back to the original tensor shape
    patches = patches.view(
        batch_size,
        channels * patch_height * patch_width,
        total_patches
    )  # Shape: (batch_size, channels * patch_h * patch_w, num_patches)

    # Fold the patches back into the original tensor shape
    modified_tensor = F.fold(
        patches,
        output_size=(height, width),
        kernel_size=patch_size,
        stride=patch_size
    )

    return modified_tensor

