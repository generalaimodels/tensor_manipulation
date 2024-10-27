import torch
import torch.nn.functional as F
from typing import Tuple
import math


def texture_based_patch_reorganization(
    input_tensor: torch.Tensor,
    patch_size: Tuple[int, int],
    epsilon: float = 0.05
) -> torch.Tensor:
    """
    Reorganize a percentage of patches in the input tensor using texture synthesis.

    This function splits each image in the batch into non-overlapping patches,
    randomly selects a specified percentage of these patches, and replaces them
    with other randomly selected patches from the same image.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        patch_size (Tuple[int, int]): Size of each patch as (patch_height, patch_width).
        epsilon (float, optional): Fraction of patches to reorganize. Must be between 0 and 1.
            Defaults to 0.05.

    Returns:
        torch.Tensor: Tensor with reorganized patches, same shape as input_tensor.

    Raises:
        ValueError: If input_tensor is not a 4D tensor.
        ValueError: If patch dimensions do not divide the input height and width.
        ValueError: If epsilon is not between 0 and 1.
    """
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError("input_tensor must be a torch.Tensor.")

    if input_tensor.dim() != 4:
        raise ValueError("input_tensor must be a 4D tensor with shape (batch_size, channels, height, width).")

    if not (0.0 <= epsilon <= 1.0):
        raise ValueError("epsilon must be a float between 0 and 1.")

    batch_size, channels, height, width = input_tensor.size()
    patch_height, patch_width = patch_size

    if height % patch_height != 0 or width % patch_width != 0:
        raise ValueError("patch_size dimensions must divide height and width of input_tensor exactly.")

    # Calculate number of patches along height and width
    num_patches_h = height // patch_height
    num_patches_w = width // patch_width
    num_patches = num_patches_h * num_patches_w

    # Unfold the input tensor to extract patches
    # Shape after unfold: (batch_size, channels * patch_height * patch_width, num_patches)
    patches = F.unfold(
        input_tensor,
        kernel_size=patch_size,
        stride=patch_size
    )

    # Determine number of patches to replace
    num_patches_to_replace = math.ceil(epsilon * num_patches)

    if num_patches_to_replace == 0:
        return input_tensor.clone()

    # Generate random indices for patches to replace
    replace_indices = torch.randint(
        low=0,
        high=num_patches,
        size=(num_patches_to_replace,),
        device=input_tensor.device
    )

    # Generate random indices for patches to sample from
    sample_indices = torch.randint(
        low=0,
        high=num_patches,
        size=(num_patches_to_replace,),
        device=input_tensor.device
    )

    # Ensure that sampled indices are different from replace indices
    # In the rare case of overlap, resample
    mask = replace_indices == sample_indices
    while mask.any():
        sample_indices[mask] = torch.randint(
            low=0,
            high=num_patches,
            size=(mask.sum(),),
            device=input_tensor.device
        )
        mask = replace_indices == sample_indices

    # Expand indices for batch processing
    # Shape: (batch_size, num_patches_to_replace)
    replace_indices_expanded = replace_indices.unsqueeze(0).expand(batch_size, -1)
    sample_indices_expanded = sample_indices.unsqueeze(0).expand(batch_size, -1)

    # Gather the patches to replace and the new patches
    # Shape after gather: (batch_size, channels * patch_height * patch_width, num_patches_to_replace)
    patches_to_replace = patches.gather(
        dim=2,
        index=replace_indices_expanded.unsqueeze(1).expand(-1, channels * patch_height * patch_width, -1)
    )
    new_patches = patches.gather(
        dim=2,
        index=sample_indices_expanded.unsqueeze(1).expand(-1, channels * patch_height * patch_width, -1)
    )

    # Replace the selected patches with new patches
    patches.scatter_(
        dim=2,
        index=replace_indices_expanded.unsqueeze(1).expand(-1, channels * patch_height * patch_width, -1),
        src=new_patches
    )

    # Fold the patches back to the original tensor shape
    output_tensor = F.fold(
        patches,
        output_size=(height, width),
        kernel_size=patch_size,
        stride=patch_size
    )

    return output_tensor

