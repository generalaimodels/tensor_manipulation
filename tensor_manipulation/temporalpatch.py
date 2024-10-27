import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def temporal_patch_mixing(
    input_tensor: Tensor,
    patch_size: Tuple[int, int],
    eps: float = 0.05,
) -> Tensor:
    """
    Perform Temporal Patch Mixing (Spatio-Temporal Transforms) on the input tensor.

    Args:
        input_tensor (Tensor): Input tensor of shape (batch_size, channels, height, width).
        patch_size (Tuple[int, int]): Size of the patches to extract (height, width).
        eps (float, optional): Epsilon value for perturbation. Defaults to 0.05.

    Returns:
        Tensor: Transformed tensor with shape (batch_size, channels, height, width).

    Raises:
        ValueError: If input tensor dimensions are incorrect or patch size is invalid.
    """
    try:
        # Validate input tensor dimensions
        if input_tensor.dim() != 4:
            raise ValueError(
                f"Expected input tensor to have 4 dimensions, got {input_tensor.dim()}."
            )

        batch_size, channels, height, width = input_tensor.shape
        patch_height, patch_width = patch_size

        if patch_height <= 0 or patch_width <= 0:
            raise ValueError("Patch dimensions must be positive integers.")

        if height % patch_height != 0 or width % patch_width != 0:
            raise ValueError(
                "Height and width must be divisible by patch dimensions."
            )

        # Initialize Unfold and Fold
        unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        fold = nn.Fold(
            output_size=(height, width), kernel_size=patch_size, stride=patch_size
        )

        # Extract patches
        patches = unfold(input_tensor)  # Shape: (batch_size, channels * patch_height * patch_width, num_patches)
        num_patches = patches.size(-1)
        patches = patches.view(batch_size, channels, patch_height, patch_width, num_patches)

        # Apply Fourier Transform to patches
        patches_fft = torch.fft.fft2(patches, dim=(-2, -1))
        patches_fft = patches_fft * (1 + eps * torch.randn_like(patches_fft))

        # Swap patches across time frames
        permuted_indices = torch.randperm(batch_size)
        patches_fft = patches_fft[permuted_indices]

        # Apply Inverse Fourier Transform
        patches_ifft = torch.fft.ifft2(patches_fft, dim=(-2, -1)).real

        # Reshape patches back
        patches_ifft = patches_ifft.view(batch_size, channels * patch_height * patch_width, num_patches)

        # Reconstruct the tensor
        output_tensor = fold(patches_ifft)

        return output_tensor

    except Exception as e:
        logger.error(f"Error in temporal_patch_mixing: {e}")
        raise

