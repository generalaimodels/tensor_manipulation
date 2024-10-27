import math
from typing import Tuple

import torch
import torch.nn.functional as F


def entropy_based_patch_swap(
    tensor: torch.Tensor,
    patch_size: Tuple[int, int],
    eps: float = 0.05,
    num_bins: int = 256,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Perform entropy-based patch swapping on a 4D tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape (B, C, H, W).
        patch_size (Tuple[int, int]): Size of each patch as (patch_height, patch_width).
        eps (float, optional): Fraction of patches to swap (default is 0.05).
        num_bins (int, optional): Number of bins for histogram calculation (default is 256).
        device (torch.device, optional): Device to perform computations on (default is CPU).

    Returns:
        torch.Tensor: Tensor after patch swapping with shape (B, C, H, W).

    Raises:
        ValueError: If input tensor dimensions are invalid or patch size does not divide H and W.
    """
    if tensor.dim() != 4:
        raise ValueError(
            f"Input tensor must be 4-dimensional, but got {tensor.dim()} dimensions."
        )

    if not isinstance(patch_size, tuple) or len(patch_size) != 2:
        raise ValueError(
            f"patch_size must be a tuple of two integers, got {patch_size}."
        )

    batch_size, channels, height, width = tensor.shape
    patch_height, patch_width = patch_size

    if height % patch_height != 0 or width % patch_width != 0:
        raise ValueError(
            f"Height ({height}) and width ({width}) must be divisible by "
            f"patch_height ({patch_height}) and patch_width ({patch_width})."
        )

    num_patches_h = height // patch_height
    num_patches_w = width // patch_width
    num_patches = num_patches_h * num_patches_w
    num_swaps = math.ceil(num_patches * eps)

    # Move tensor to the specified device
    tensor = tensor.to(device)

    # Extract patches using unfold
    patches = F.unfold(
        tensor, kernel_size=patch_size, stride=patch_size
    )  # Shape: (B, C * P_H * P_W, N)
    patches = patches.permute(0, 2, 1).contiguous()  # Shape: (B, N, C * P_H * P_W)

    # Reshape for entropy computation
    patches_reshaped = patches.view(
        batch_size,
        num_patches,
        channels,
        patch_height,
        patch_width,
    )  # Shape: (B, N, C, P_H, P_W)

    # Compute entropy for each patch
    entropies = compute_entropy(patches_reshaped, num_bins)

    # Perform swapping based on entropy similarity
    swapped_patches = swap_patches_based_on_entropy(
        patches, entropies, num_swaps, device
    )

    # Reshape back to original tensor shape
    swapped_patches = swapped_patches.permute(0, 2, 1).contiguous()  # Shape: (B, C * P_H * P_W, N)
    swapped_tensor = F.fold(
        swapped_patches,
        output_size=(height, width),
        kernel_size=patch_size,
        stride=patch_size,
    )

    return swapped_tensor


def compute_entropy(
    patches: torch.Tensor, num_bins: int
) -> torch.Tensor:
    """
    Compute entropy for each patch using histogram-based estimation.

    Args:
        patches (torch.Tensor): Patches tensor of shape (B, N, C, P_H, P_W).
        num_bins (int): Number of bins for histogram.

    Returns:
        torch.Tensor: Entropy tensor of shape (B, N).
    """
    batch_size, num_patches, channels, patch_h, patch_w = patches.shape
    patches_flat = patches.reshape(batch_size, num_patches, -1)  # Shape: (B, N, C*P_H*P_W)

    # Normalize pixel values to [0, 1]
    min_vals = patches_flat.min(dim=2, keepdim=True).values
    max_vals = patches_flat.max(dim=2, keepdim=True).values
    patches_normalized = (patches_flat - min_vals) / (
        max_vals - min_vals + 1e-8
    )  # Shape: (B, N, C*P_H*P_W)

    # Compute bin size
    bin_size = 1.0 / num_bins

    # Compute bin indices
    bin_indices = torch.clamp(
        (patches_normalized / bin_size).long(), 0, num_bins - 1
    )  # Shape: (B, N, C*P_H*P_W)

    # Reshape for scatter_add
    bin_indices = bin_indices.reshape(batch_size * num_patches, -1)  # Shape: (B*N, C*P_H*P_W)
    patches_flat = patches_flat.reshape(batch_size * num_patches, -1)  # Shape: (B*N, C*P_H*P_W)

    # Initialize histogram
    hist = torch.zeros(
        batch_size * num_patches, num_bins, device=patches.device
    )  # Shape: (B*N, num_bins)

    # Compute histograms using scatter_add
    hist.scatter_add_(1, bin_indices, torch.ones_like(bin_indices, dtype=torch.float))

    # Convert histograms to probabilities
    prob = hist / (hist.sum(dim=1, keepdim=True) + 1e-8)  # Shape: (B*N, num_bins)

    # Compute entropy
    entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=1)  # Shape: (B*N,)
    entropy = entropy.reshape(batch_size, num_patches)  # Shape: (B, N)

    return entropy


def swap_patches_based_on_entropy(
    patches: torch.Tensor,
    entropies: torch.Tensor,
    num_swaps: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Swap patches with similar entropy values to modify texture consistency.

    Args:
        patches (torch.Tensor): Patches tensor of shape (B, N, C * P_H * P_W).
        entropies (torch.Tensor): Entropy tensor of shape (B, N).
        num_swaps (int): Number of patches to swap.
        device (torch.device, optional): Device to perform computations on (default is CPU).

    Returns:
        torch.Tensor: Swapped patches tensor of shape (B, N, C * P_H * P_W).
    """
    batch_size, num_patches, _ = patches.shape
    swapped_patches = patches.clone()

    for b in range(batch_size):
        entropy = entropies[b]
        # Sort patches based on entropy
        sorted_indices = torch.argsort(entropy)
        
        # Determine number of swap pairs
        num_pairs = num_swaps // 2

        # Avoid exceeding available indices
        num_pairs = min(num_pairs, num_patches // 2)

        for i in range(num_pairs):
            idx1 = sorted_indices[i].item()
            idx2 = sorted_indices[-(i + 1)].item()

            if idx1 != idx2:
                # Swap patches
                temp = swapped_patches[b, idx1].clone()
                swapped_patches[b, idx1] = swapped_patches[b, idx2]
                swapped_patches[b, idx2] = temp

    return swapped_patches


