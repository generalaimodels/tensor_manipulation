import torch
import torch.nn.functional as F
from typing import Tuple
import math


def add_random_gaussian_noise_to_patches(
    input_tensor: torch.Tensor,
    patch_size: Tuple[int, int],
    noise_eps: float = 0.05,
    noise_fraction: float = 0.05,
    seed: int = None
) -> torch.Tensor:
    """
    Adds Gaussian noise to a random subset of patches in the input tensor.

    Args:
        input_tensor (torch.Tensor): A tensor of shape (batch_size, channels, height, width).
        patch_size (Tuple[int, int]): The size of each patch as (patch_height, patch_width).
        noise_eps (float, optional): The standard deviation of the Gaussian noise. Defaults to 0.05.
        noise_fraction (float, optional): The fraction of patches to which noise will be added. Defaults to 0.05.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        torch.Tensor: The tensor with Gaussian noise added to a subset of patches, maintaining the original shape.

    Raises:
        ValueError: If the input tensor dimensions are incompatible with the patch size.
        TypeError: If the input tensor is not a 4-dimensional torch.Tensor.
    """
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError("input_tensor must be a torch.Tensor.")
    if input_tensor.dim() != 4:
        raise ValueError("input_tensor must be a 4-dimensional tensor (batch_size, channels, height, width).")

    batch_size, channels, height, width = input_tensor.shape
    patch_height, patch_width = patch_size

    if height % patch_height != 0 or width % patch_width != 0:
        raise ValueError(
            "Height and width of the input tensor must be divisible by the patch size."
        )

    num_patches_h = height // patch_height
    num_patches_w = width // patch_width
    total_patches = batch_size * num_patches_h * num_patches_w

    # Set random seed if provided for reproducibility
    if seed is not None:
        torch.manual_seed(seed)

    # Determine number of patches to add noise to
    num_noisy_patches = math.ceil(total_patches * noise_fraction)

    # Generate unique random indices for patches to add noise
    noisy_patch_indices = torch.randperm(total_patches)[:num_noisy_patches]

    # Reshape input tensor to extract patches
    patches = input_tensor.unfold(2, patch_height, patch_height).unfold(3, patch_width, patch_width)
    # patches shape: (batch_size, channels, num_patches_h, num_patches_w, patch_height, patch_width)
    patches = patches.contiguous().view(batch_size, channels, -1, patch_height, patch_width)
    # patches shape: (batch_size, channels, total_patches_per_batch, patch_height, patch_width)

    # Reshape to (batch_size * total_patches_per_batch, channels, patch_height, patch_width)
    patches = patches.view(-1, channels, patch_height, patch_width)

    # Create a mask for patches to add noise
    noise_mask = torch.zeros(patches.size(0), dtype=torch.bool)
    noise_mask[noisy_patch_indices] = True

    # Generate Gaussian noise
    noise = torch.randn_like(patches) * noise_eps

    # Add noise to selected patches
    patches[noise_mask] += noise[noise_mask]

    # Reshape patches back to original tensor shape
    patches = patches.view(batch_size, num_patches_h, num_patches_w, channels, patch_height, patch_width)
    patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
    # patches shape: (batch_size, channels, num_patches_h, patch_height, num_patches_w, patch_width)

    # Fold patches back into the original tensor shape
    output_tensor = patches.view(
        batch_size,
        channels,
        num_patches_h * patch_height,
        num_patches_w * patch_width
    )

    return output_tensor

