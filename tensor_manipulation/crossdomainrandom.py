import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Callable
import math
import random


def cross_domain_random_permutation(
    tensor: torch.Tensor,
    patch_size: Tuple[int, int],
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    epsilon: float = 0.05,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Apply Cross-Domain Random Permutation on input tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape (B, C, H, W).
        patch_size (Tuple[int, int]): Size of each patch (patch_height, patch_width).
        transform (Optional[Callable[[torch.Tensor], torch.Tensor]]): 
            Transformation function to apply to selected patches.
        epsilon (float, optional): Fraction of patches to permute. Default is 0.05.
        seed (Optional[int], optional): Random seed for reproducibility. Default is None.

    Returns:
        torch.Tensor: Tensor after applying cross-domain random permutation, 
                      with shape (B, C, H, W).

    Raises:
        ValueError: If input tensor dimensions are invalid or patch size is incompatible.
    """
    if tensor.ndim != 4:
        raise ValueError(f"Input tensor must have 4 dimensions (B, C, H, W), got {tensor.ndim} dimensions.")

    if not (0 < epsilon <= 1):
        raise ValueError(f"Epsilon must be in the range (0, 1], got {epsilon}.")

    B, C, H, W = tensor.shape
    patch_h, patch_w = patch_size

    if H % patch_h != 0 or W % patch_w != 0:
        raise ValueError(f"Patch size {patch_size} must evenly divide tensor dimensions ({H}, {W}).")

    if patch_h <= 0 or patch_w <= 0:
        raise ValueError(f"Patch dimensions must be positive integers, got {patch_size}.")

    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    # Calculate number of patches
    num_patches_h = H // patch_h
    num_patches_w = W // patch_w
    num_patches = num_patches_h * num_patches_w

    # Extract patches using unfold
    patches = tensor.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
    # patches shape: (B, C, num_patches_h, num_patches_w, patch_h, patch_w)
    patches = patches.contiguous().view(B, C, num_patches, patch_h, patch_w)
    # patches shape: (B, C, N, patch_h, patch_w)

    # Determine number of patches to permute
    num_perm_patches = max(1, math.floor(epsilon * num_patches))

    # Perform permutation for each image in the batch
    for b in range(B):
        # Select patches to permute
        perm_indices = random.sample(range(num_patches), num_perm_patches)
        
        # Apply transformation if provided
        if transform is not None:
            selected_patches = patches[b, :, perm_indices, :, :]  # Shape: (C, N, H_p, W_p)
            # Reshape to (N, C, H_p, W_p) for transformation
            selected_patches = selected_patches.permute(1, 0, 2, 3).contiguous()
            # Apply transformation
            transformed_patches = transform(selected_patches)
            if transformed_patches.shape != selected_patches.shape:
                raise ValueError(
                    f"Transform function must return tensor of shape {selected_patches.shape}, "
                    f"but got {transformed_patches.shape}."
                )
            # Reshape back to (C, N, H_p, W_p)
            transformed_patches = transformed_patches.permute(1, 0, 2, 3).contiguous()
            # Assign back
            patches[b, :, perm_indices, :, :] = transformed_patches

        # Permute patches across channels
        for c in range(C):
            shuffled_indices = perm_indices.copy()
            random.shuffle(shuffled_indices)
            patches[b, c, perm_indices, :, :] = patches[b, c, shuffled_indices, :, :]

    # Reconstruct the tensor from patches
    patches = patches.view(B, C, num_patches_h, num_patches_w, patch_h, patch_w)
    patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
    tensor_reconstructed = patches.view(B, C, H, W)

    return tensor_reconstructed

