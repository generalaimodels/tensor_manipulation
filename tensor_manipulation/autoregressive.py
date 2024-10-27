import torch
import torch.nn.functional as F
from typing import Tuple
import math


def autoregressive_patch_prediction(
    input_tensor: torch.Tensor,
    patch_size: Tuple[int, int],
    epsilon: float = 0.05,
    debug: bool = False
) -> torch.Tensor:
    """
    Performs autoregressive patch prediction on the input tensor.

    Treats the tensor as composed of patches and predicts each patch's values based
    on the mean and variance of its neighboring patches to enforce statistical
    consistency within a specified epsilon.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (B, C, H, W).
        patch_size (Tuple[int, int]): Size of each patch as (patch_height, patch_width).
        epsilon (float, optional): Tolerance for statistical consistency. Defaults to 0.05.
        debug (bool, optional): If True, prints debug information. Defaults to False.

    Returns:
        torch.Tensor: Tensor after autoregressive patch prediction with shape (B, C, H, W).

    Raises:
        TypeError: If input_tensor is not a torch.Tensor or patch_size is not a tuple of integers.
        ValueError: If input_tensor is not 4D, patch_size dimensions are not positive integers,
                    or epsilon is not between 0 and 1.
    """
    # Validate input_tensor dimensions
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError("input_tensor must be a torch.Tensor.")
    if input_tensor.dim() != 4:
        raise ValueError(
            f"input_tensor must be a 4D tensor, but got {input_tensor.dim()}D tensor."
        )

    B, C, H, W = input_tensor.shape
    patch_h, patch_w = patch_size

    # Validate patch_size
    if not (isinstance(patch_h, int) and isinstance(patch_w, int)):
        raise TypeError("patch_size must be a tuple of two integers.")
    if patch_h <= 0 or patch_w <= 0:
        raise ValueError("patch_size dimensions must be positive integers.")

    # Validate epsilon
    if not (0 < epsilon < 1):
        raise ValueError("epsilon must be a float between 0 and 1.")

    if debug:
        print(f"Input Tensor Shape: {input_tensor.shape}")

    # Calculate number of patches
    num_patches_h = math.ceil(H / patch_h)
    num_patches_w = math.ceil(W / patch_w)

    # Pad input tensor to make it divisible by patch size
    pad_h = num_patches_h * patch_h - H
    pad_w = num_patches_w * patch_w - W
    padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
    input_padded = F.pad(input_tensor, padding, mode='reflect')

    if debug:
        print(f"Padded Tensor Shape: {input_padded.shape}")

    # Reshape to patches
    patches = input_padded.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
    # patches shape: (B, C, num_patches_h, num_patches_w, patch_h, patch_w)

    if debug:
        print(f"Patches Shape: {patches.shape}")

    # Compute mean and variance for each patch
    patch_means = patches.mean(dim=(-1, -2))  # (B, C, num_patches_h, num_patches_w)
    patch_vars = patches.var(dim=(-1, -2), unbiased=False)  # (B, C, num_patches_h, num_patches_w)

    if debug:
        print(f"Patch Means Shape: {patch_means.shape}")
        print(f"Patch Vars Shape: {patch_vars.shape}")

    # Initialize predicted patches with original patches
    predicted_patches = patches.clone()

    # Define neighbor offsets (up, down, left, right)
    neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i in range(num_patches_h):
        for j in range(num_patches_w):
            neighbors_mean = []
            neighbors_var = []

            for di, dj in neighbor_offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < num_patches_h and 0 <= nj < num_patches_w:
                    neighbors_mean.append(patch_means[:, :, ni, nj])
                    neighbors_var.append(patch_vars[:, :, ni, nj])

            if neighbors_mean:
                # Stack neighbors' statistics
                neighbors_mean = torch.stack(neighbors_mean, dim=-1)  # (B, C, num_neighbors)
                neighbors_var = torch.stack(neighbors_var, dim=-1)    # (B, C, num_neighbors)

                # Compute aggregated statistics
                agg_mean = neighbors_mean.mean(dim=-1, keepdim=True)  # (B, C, 1)
                agg_var = neighbors_var.mean(dim=-1, keepdim=True)    # (B, C, 1)

                if debug:
                    print(f"Patches [{i}, {j}] - Aggregated Mean Shape: {agg_mean.shape}")
                    print(f"Patches [{i}, {j}] - Aggregated Var Shape: {agg_var.shape}")

                # Predict current patch based on aggregated statistics
                predicted_mean = agg_mean  # (B, C, 1)
                predicted_var = agg_var    # (B, C, 1)

                # Generate predicted patch with mean and variance
                std = torch.sqrt(predicted_var + 1e-8)  # (B, C, 1)
                noise = torch.randn_like(patches[:, :, i, j, :, :])  # (B, C, patch_h, patch_w)

                if debug:
                    print(f"Patches [{i}, {j}] - Generating noise with shape: {noise.shape}")
                    print(f"Patches [{i}, {j}] - STD shape: {std.shape}")
                    print(f"Patches [{i}, {j}] - Predicted Mean shape: {predicted_mean.shape}")

                # **Corrected Line**: Removed one unsqueeze to avoid extra dimension
                predicted_patch = predicted_mean.unsqueeze(-1) + std.unsqueeze(-1) * noise
                # Shape: (B, C, patch_h, patch_w)

                if debug:
                    print(f"Patches [{i}, {j}] - Predicted Patch Shape: {predicted_patch.shape}")

                # Ensure statistical consistency within epsilon
                pred_patch_mean = predicted_patch.mean(dim=(-1, -2), keepdim=True)  # (B, C, 1, 1)
                pred_patch_var = predicted_patch.var(dim=(-1, -2), unbiased=False, keepdim=True)  # (B, C, 1, 1)

                mean_diff = torch.abs(pred_patch_mean - agg_mean.unsqueeze(-1)) / (agg_mean.unsqueeze(-1) + 1e-8)
                var_diff = torch.abs(pred_patch_var - agg_var.unsqueeze(-1)) / (agg_var.unsqueeze(-1) + 1e-8)

                mask = (mean_diff <= epsilon) & (var_diff <= epsilon)  # (B, C, 1, 1)

                if debug:
                    print(f"Patches [{i}, {j}] - Mask Shape Before Aggregation: {mask.shape}")

                # **Ensuring Mask Shape**: Remove extra dimensions if present
                if mask.dim() > 4:
                    # If mask has shape [B, C, num_neighbors, 1, 1], aggregate across num_neighbors
                    mask = mask.all(dim=2, keepdim=False).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
                    if debug:
                        print(f"Patches [{i}, {j}] - Mask Shape After Aggregation: {mask.shape}")

                # Apply mask to decide whether to replace the patch
                predicted_patches[:, :, i, j, :, :] = torch.where(
                    mask,  # [B, C, 1, 1] - broadcasts to [B, C, patch_h, patch_w]
                    predicted_patch,  # [B, C, patch_h, patch_w]
                    patches[:, :, i, j, :, :]  # [B, C, patch_h, patch_w]
                )

                if debug:
                    print(f"Patches [{i}, {j}] - Patch Updated.\n")

    # Reshape patches back to the original tensor shape
    output_padded = predicted_patches.contiguous().view(
        B, C, num_patches_h * patch_h, num_patches_w * patch_w
    )

    if debug:
        print(f"Output Padded Shape: {output_padded.shape}")

    # Remove padding
    output = output_padded[:, :, :H, :W]

    if debug:
        print(f"Output Tensor Shape: {output.shape}")

    return output

