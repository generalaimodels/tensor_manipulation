import numpy as np
import torch
from typing import Tuple
import random


def mrf_patch_stitching(
    tensor: torch.Tensor,
    patch_size: Tuple[int, int],
    epsilon: float = 0.05
) -> torch.Tensor:
    """
    Perform Markov Random Field Patch Stitching on the input tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        patch_size (Tuple[int, int]): Size of each patch (patch_height, patch_width).
        epsilon (float, optional): Fraction of patches to infer. Defaults to 0.05.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, channels, height, width).
    """
    try:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input tensor must be a torch.Tensor.")
        if tensor.dim() != 4:
            raise ValueError("Input tensor must have 4 dimensions (batch_size, channels, height, width).")
        if not (0 < epsilon < 1):
            raise ValueError("Epsilon must be a float between 0 and 1.")
        if not isinstance(patch_size, Tuple) or len(patch_size) != 2:
            raise ValueError("patch_size must be a tuple of two integers.")

        batch_size, channels, height, width = tensor.shape
        patch_height, patch_width = patch_size

        if height % patch_height != 0 or width % patch_width != 0:
            raise ValueError("Height and width must be divisible by patch_height and patch_width respectively.")

        num_patches_h = height // patch_height
        num_patches_w = width // patch_width
        total_patches = num_patches_h * num_patches_w

        # Reshape tensor into patches
        patches = tensor.unfold(2, patch_height, patch_height).unfold(3, patch_width, patch_width)
        # patches shape: (batch_size, channels, num_patches_h, num_patches_w, patch_height, patch_width)
        patches = patches.contiguous().view(batch_size, channels, total_patches, patch_height, patch_width)

        # Select 5% of patches to infer
        num_patches_to_infer = max(1, int(total_patches * epsilon))
        patch_indices = list(range(total_patches))
        infer_indices = random.sample(patch_indices, num_patches_to_infer)

        # Iterate over batches
        for b in range(batch_size):
            for c in range(channels):
                for idx in infer_indices:
                    # Find neighbors: left, right, top, bottom
                    row = idx // num_patches_w
                    col = idx % num_patches_w
                    neighbors = []

                    if row > 0:
                        neighbors.append(idx - num_patches_w)
                    if row < num_patches_h - 1:
                        neighbors.append(idx + num_patches_w)
                    if col > 0:
                        neighbors.append(idx - 1)
                    if col < num_patches_w - 1:
                        neighbors.append(idx + 1)

                    if not neighbors:
                        continue  # Skip if no neighbors

                    # Average the values of neighboring patches
                    neighbor_values = [patches[b, c, n].mean() for n in neighbors]
                    mean_value = sum(neighbor_values) / len(neighbor_values)

                    # Update the patch with the mean value
                    patches[b, c, idx] = torch.full_like(patches[b, c, idx], mean_value)

        # Reconstruct the tensor from patches
        # First, reshape patches back to (batch_size, channels, num_patches_h, num_patches_w, patch_height, patch_width)
        patches = patches.view(batch_size, channels, num_patches_h, num_patches_w, patch_height, patch_width)
        # Permute to (batch_size, channels, num_patches_h, patch_height, num_patches_w, patch_width)
        patches = patches.permute(0, 1, 2, 4, 3, 5)
        # Reshape to (batch_size, channels, height, width)
        stitched_tensor = patches.contiguous().view(batch_size, channels, height, width)

        return stitched_tensor

    except Exception as e:
        raise RuntimeError(f"An error occurred during MRF Patch Stitching: {str(e)}")
