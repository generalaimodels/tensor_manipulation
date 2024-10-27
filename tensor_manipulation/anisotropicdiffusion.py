import torch
import torch.nn.functional as F
from typing import Tuple
import math
import random


def anisotropic_diffusion_patch_filtering(
    tensor: torch.Tensor,
    patch_size: Tuple[int, int],
    eps: float = 0.05,
    num_iterations: int = 10,
    k: float = 20.0,
) -> torch.Tensor:
    """
    Apply anisotropic diffusion (edge-preserving) filtering to 5% of tensor patches.

    Args:
        tensor (torch.Tensor): Input tensor with shape (B, C, H, W).
        patch_size (Tuple[int, int]): The size of each patch as (patch_height, patch_width).
        eps (float, optional): Fraction of patches to filter. Defaults to 0.05.
        num_iterations (int, optional): Number of diffusion iterations. Defaults to 10.
        k (float, optional): Conduction coefficient for diffusion. Defaults to 20.0.

    Returns:
        torch.Tensor: Output tensor with the same shape as the input (B, C, H, W).

    Raises:
        ValueError: If tensor dimensions are incompatible with patch size.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input tensor must be a torch.Tensor.")

    if tensor.dim() != 4:
        raise ValueError(
            f"Input tensor must have 4 dimensions (B, C, H, W), got {tensor.dim()}."
        )

    batch_size, channels, height, width = tensor.shape
    patch_height, patch_width = patch_size

    if height % patch_height != 0 or width % patch_width != 0:
        raise ValueError(
            "Height and width of the tensor must be divisible by patch dimensions."
        )

    # Calculate the number of patches along height and width
    num_patches_h = height // patch_height
    num_patches_w = width // patch_width
    num_patches = num_patches_h * num_patches_w

    # Determine the number of patches to filter (5%)
    num_filter_patches = max(1, math.ceil(eps * num_patches))

    # Reshape tensor to (B, C, num_patches, patch_height, patch_width)
    tensor_patches = tensor.unfold(
        dimension=2, size=patch_height, step=patch_height
    ).unfold(dimension=3, size=patch_width, step=patch_width)
    tensor_patches = tensor_patches.contiguous().view(
        batch_size, channels, num_patches, patch_height, patch_width
    )

    # Select random patch indices to filter for each sample in the batch
    selected_indices = []
    for _ in range(batch_size):
        indices = random.sample(range(num_patches), num_filter_patches)
        selected_indices.append(indices)

    # Apply anisotropic diffusion to selected patches
    for b in range(batch_size):
        for c in range(channels):
            for idx in selected_indices[b]:
                patch = tensor_patches[b, c, idx].clone()
                patch = anisotropic_diffusion(
                    patch, num_iterations=num_iterations, k=k
                )
                tensor_patches[b, c, idx] = patch

    # Reshape back to the original tensor shape
    tensor_patches = tensor_patches.view(
        batch_size,
        channels,
        num_patches_h,
        num_patches_w,
        patch_height,
        patch_width,
    )
    tensor_patches = tensor_patches.permute(0, 1, 2, 4, 3, 5).contiguous()
    output_tensor = tensor_patches.view(batch_size, channels, height, width)

    return output_tensor


def anisotropic_diffusion(
    patch: torch.Tensor,
    num_iterations: int = 10,
    k: float = 20.0,
    lambda_coef: float = 0.25,
) -> torch.Tensor:
    """
    Perform Perona-Malik anisotropic diffusion on a single patch.

    Args:
        patch (torch.Tensor): 2D tensor representing a single image patch.
        num_iterations (int, optional): Number of diffusion iterations. Defaults to 10.
        k (float, optional): Conduction coefficient. Controls sensitivity to edges.
        lambda_coef (float, optional): Time step coefficient. Must be <= 0.25 for stability.

    Returns:
        torch.Tensor: Diffused patch.

    Raises:
        ValueError: If lambda coefficient is greater than 0.25.
    """
    if lambda_coef > 0.25:
        raise ValueError("Lambda coefficient must be <= 0.25 for stability.")

    patch = patch.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)

    for _ in range(num_iterations):
        # Compute gradients
        north = F.pad(patch[:, :, 1:, :], (0, 0, 0, 1))
        south = F.pad(patch[:, :, :-1, :], (0, 0, 1, 0))
        east = F.pad(patch[:, :, :, :-1], (1, 0, 0, 0))
        west = F.pad(patch[:, :, :, 1:], (0, 1, 0, 0))

        delta_north = north - patch
        delta_south = south - patch
        delta_east = east - patch
        delta_west = west - patch

        # Compute conduction coefficients
        c_north = torch.exp(-(delta_north / k) ** 2)
        c_south = torch.exp(-(delta_south / k) ** 2)
        c_east = torch.exp(-(delta_east / k) ** 2)
        c_west = torch.exp(-(delta_west / k) ** 2)

        # Update the patch
        diffusion = (
            c_north * delta_north
            + c_south * delta_south
            + c_east * delta_east
            + c_west * delta_west
        )
        patch = patch + lambda_coef * diffusion

    return patch.squeeze()


