import torch
import torch.nn.functional as F
from typing import Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def kurtosis_based_patch_selection(
    tensor: torch.Tensor,
    patch_size: Tuple[int, int],
    eps: float = 0.05
) -> torch.Tensor:
    """
    Select and augment patches based on the kurtosis of their intensity distribution.

    Args:
        tensor (torch.Tensor): Input tensor with shape (batch_size, channels, height, width).
        patch_size (Tuple[int, int]): Size of each patch as (patch_height, patch_width).
        eps (float, optional): Percentage (0 < eps < 1) determining the percentile for selection.
                               Default is 0.05 (5%).

    Returns:
        torch.Tensor: Tensor with augmented patches, maintaining the original shape (b, c, H, W).

    Raises:
        ValueError: If input tensor does not have 4 dimensions.
        ValueError: If patch_size dimensions are larger than tensor spatial dimensions.
        ValueError: If eps is not between 0 and 0.5.
    """
    # Validate input tensor dimensions
    if tensor.dim() != 4:
        logger.error("Input tensor must be 4-dimensional (batch_size, channels, height, width).")
        raise ValueError("Input tensor must be 4-dimensional (batch_size, channels, height, width).")

    batch_size, channels, height, width = tensor.shape
    patch_height, patch_width = patch_size

    # Validate patch size
    if patch_height <= 0 or patch_width <= 0:
        logger.error("Patch dimensions must be positive integers.")
        raise ValueError("Patch dimensions must be positive integers.")
    if patch_height > height or patch_width > width:
        logger.error("Patch size cannot be larger than tensor spatial dimensions.")
        raise ValueError("Patch size cannot be larger than tensor spatial dimensions.")

    # Validate eps
    if not (0 < eps < 0.5):
        logger.error("Parameter eps must be between 0 and 0.5.")
        raise ValueError("Parameter eps must be between 0 and 0.5.")

    # Calculate the number of patches along height and width
    num_patches_h = height // patch_height
    num_patches_w = width // patch_width

    if num_patches_h == 0 or num_patches_w == 0:
        logger.error("Patch size is too large for the given tensor dimensions.")
        raise ValueError("Patch size is too large for the given tensor dimensions.")

    # Adjust tensor dimensions to fit an integer number of patches
    trimmed_height = num_patches_h * patch_height
    trimmed_width = num_patches_w * patch_width
    tensor_trimmed = tensor[:, :, :trimmed_height, :trimmed_width]

    # Reshape tensor to extract patches
    patches = tensor_trimmed.unfold(2, patch_height, patch_height).unfold(3, patch_width, patch_width)
    # patches shape: (batch_size, channels, num_patches_h, num_patches_w, patch_height, patch_width)

    # Compute mean and variance for each patch
    patches_flat = patches.contiguous().view(batch_size, channels, num_patches_h, num_patches_w, -1)
    mean = patches_flat.mean(dim=-1, keepdim=True)
    variance = patches_flat.var(dim=-1, unbiased=False, keepdim=True)

    # Compute kurtosis: (E[(X - mu)^4]) / (variance^2) - 3
    fourth_moment = ((patches_flat - mean) ** 4).mean(dim=-1, keepdim=True)
    kurtosis = (fourth_moment / (variance ** 2)) - 3  # Shape: (batch_size, channels, num_patches_h, num_patches_w, 1)

    # Aggregate kurtosis across channels by averaging
    kurtosis_avg = kurtosis.mean(dim=1)  # Shape: (batch_size, num_patches_h, num_patches_w, 1)

    # Squeeze the last dimension
    kurtosis_avg = kurtosis_avg.squeeze(-1)  # Shape: (batch_size, num_patches_h, num_patches_w)

    # Flatten the patches for each batch to compute quantiles
    kurtosis_avg_flat = kurtosis_avg.view(batch_size, -1)  # Shape: (batch_size, num_patches_h * num_patches_w)

    # Determine thresholds for high and low kurtosis based on eps
    high_threshold = torch.quantile(kurtosis_avg_flat, 1 - eps, dim=1, keepdim=True)  # Shape: (batch_size, 1)
    low_threshold = torch.quantile(kurtosis_avg_flat, eps, dim=1, keepdim=True)       # Shape: (batch_size, 1)

    # Reshape thresholds for broadcasting
    high_threshold = high_threshold.view(batch_size, 1, 1)  # Shape: (batch_size, 1, 1)
    low_threshold = low_threshold.view(batch_size, 1, 1)    # Shape: (batch_size, 1, 1)

    # Create masks for high and low kurtosis patches
    high_kurtosis_mask = kurtosis_avg >= high_threshold
    low_kurtosis_mask = kurtosis_avg <= low_threshold

    # Combine masks
    selection_mask = high_kurtosis_mask | low_kurtosis_mask  # Shape: (batch_size, num_patches_h, num_patches_w)

    # Expand mask to match patch dimensions
    selection_mask = selection_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, 1, num_patches_h, num_patches_w, 1, 1)
    selection_mask = selection_mask.expand(-1, channels, -1, -1, patch_height, patch_width)  # Shape: (batch_size, channels, num_patches_h, num_patches_w, patch_height, patch_width)

    # Apply augmentation: flip selected patches horizontally
    patches_augmented = torch.where(
        selection_mask,
        patches.flip(-1),  # Horizontal flip
        patches
    )

    # Reshape back to the original tensor shape
    patches_augmented = patches_augmented.view(batch_size, channels, num_patches_h, num_patches_w, patch_height, patch_width)
    tensor_augmented = tensor_trimmed.clone()
    tensor_augmented[:, :, :trimmed_height, :trimmed_width] = patches_augmented.permute(0, 1, 2, 4, 3, 5).contiguous().view(batch_size, channels, trimmed_height, trimmed_width)

    # If trimming was applied, pad the tensor back to original size
    if trimmed_height != height or trimmed_width != width:
        pad_height = height - trimmed_height
        pad_width = width - trimmed_width
        tensor_padded = F.pad(tensor_augmented, (0, pad_width, 0, pad_height))
        return tensor_padded
    else:
        return tensor_augmented
