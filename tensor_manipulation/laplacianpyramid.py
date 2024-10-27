import torch
import torch.nn.functional as F
from typing import Tuple, List


def build_gaussian_pyramid(tensor: torch.Tensor, levels: int) -> List[torch.Tensor]:
    """
    Constructs a Gaussian pyramid from the input tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape (B, C, H, W).
        levels (int): Number of levels in the pyramid.

    Returns:
        List[torch.Tensor]: List of tensors representing the Gaussian pyramid.
    """
    pyramid = [tensor]
    current = tensor
    for _ in range(levels - 1):
        current = F.avg_pool2d(current, kernel_size=2, stride=2)
        pyramid.append(current)
    return pyramid


def build_laplacian_pyramid(gaussian_pyramid: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Constructs a Laplacian pyramid from the Gaussian pyramid.

    Args:
        gaussian_pyramid (List[torch.Tensor]): Gaussian pyramid.

    Returns:
        List[torch.Tensor]: Laplacian pyramid.
    """
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        expanded = F.interpolate(gaussian_pyramid[i + 1], scale_factor=2, mode='bilinear', align_corners=False)
        # Ensure the expanded tensor has the same size as the current level
        expanded = F.pad(expanded, (
            0, gaussian_pyramid[i].shape[-1] - expanded.shape[-1],
            0, gaussian_pyramid[i].shape[-2] - expanded.shape[-2]
        ))
        laplacian = gaussian_pyramid[i] - expanded
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid


def reconstruct_from_pyramid(laplacian_pyramid: List[torch.Tensor]) -> torch.Tensor:
    """
    Reconstructs the tensor from its Laplacian pyramid.

    Args:
        laplacian_pyramid (List[torch.Tensor]): Laplacian pyramid.

    Returns:
        torch.Tensor: Reconstructed tensor.
    """
    reconstructed = laplacian_pyramid[-1]
    for level in reversed(laplacian_pyramid[:-1]):
        reconstructed = F.interpolate(reconstructed, scale_factor=2, mode='bilinear', align_corners=False)
        # Ensure the reconstructed tensor has the same size as the current level
        reconstructed = F.pad(reconstructed, (
            0, level.shape[-1] - reconstructed.shape[-1],
            0, level.shape[-2] - reconstructed.shape[-2]
        ))
        reconstructed = reconstructed + level
    return reconstructed


def laplacian_pyramid_blending(
    tensor: torch.Tensor,
    patch_size: Tuple[int, int],
    eps: float = 0.05,
    levels: int = 4
) -> torch.Tensor:
    """
    Performs Laplacian Pyramid Blending on the input tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape (B, C, H, W).
        patch_size (Tuple[int, int]): Size of the patches (height, width).
        eps (float, optional): Blending factor. Defaults to 0.05.
        levels (int, optional): Number of pyramid levels. Defaults to 4.

    Returns:
        torch.Tensor: Blended tensor of shape (B, C, H, W).

    Raises:
        ValueError: If input tensor does not have 4 dimensions.
        ValueError: If patch_size is not a tuple of two positive integers.
        ValueError: If eps is not between 0 and 1.
    """
    if tensor.ndim != 4:
        raise ValueError(f"Input tensor must have 4 dimensions, got {tensor.ndim}")

    if (
        not isinstance(patch_size, tuple) or
        len(patch_size) != 2 or
        not all(isinstance(dim, int) and dim > 0 for dim in patch_size)
    ):
        raise ValueError("patch_size must be a tuple of two positive integers")

    if not (0.0 < eps < 1.0):
        raise ValueError("eps must be a float between 0 and 1")

    batch_size, channels, height, width = tensor.shape
    patch_height, patch_width = patch_size

    if patch_height > height or patch_width > width:
        raise ValueError("patch_size must be smaller than the tensor dimensions")

    # Initialize empty tensor for blended result
    blended = torch.zeros_like(tensor)

    # Iterate over each patch
    for i in range(0, height, patch_height):
        for j in range(0, width, patch_width):
            # Define the patch region
            patch = tensor[:, :, i:i + patch_height, j:j + patch_width]

            # Handle boundary conditions
            bh, bw = patch.shape[2], patch.shape[3]
            if bh != patch_height or bw != patch_width:
                padding = (0, patch_width - bw if bw < patch_width else 0,
                           0, patch_height - bh if bh < patch_height else 0)
                patch = F.pad(patch, padding, mode='reflect')

            # Build Gaussian and Laplacian pyramids
            gaussian_pyr = build_gaussian_pyramid(patch, levels)
            laplacian_pyr = build_laplacian_pyramid(gaussian_pyr)

            # Blend each level with neighboring patches (example blending)
            # Here, we simply scale the Laplacian levels by eps
            blended_pyr = [lvl * (1 - eps) for lvl in laplacian_pyr]

            # Reconstruct the blended patch
            reconstructed_patch = reconstruct_from_pyramid(blended_pyr)

            # Insert the reconstructed patch back into the blended tensor
            blended[:, :, i:i + patch_height, j:j + patch_width] = reconstructed_patch[:, :, :bh, :bw]

    return blended

