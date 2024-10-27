import math
import random
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal


def dwt2d_patch(patch: np.ndarray, wavelet: str = 'db1', level: int = 1) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Perform a multi-level 2D Discrete Wavelet Transform on a single patch.

    Args:
        patch (np.ndarray): 2D input array representing the patch.
        wavelet (str, optional): Type of wavelet to use. Defaults to 'db1'.
        level (int, optional): Number of decomposition levels. Defaults to 1.

    Returns:
        List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]: List of coefficients at each level.
    """
    coeffs = []
    current = patch.copy()
    for _ in range(level):
        # Define db1 wavelet filters
        if wavelet == 'db1':
            low_filter = np.array([1 / math.sqrt(2), 1 / math.sqrt(2)])
            high_filter = np.array([1 / math.sqrt(2), -1 / math.sqrt(2)])
        else:
            raise ValueError(f"Unsupported wavelet type: {wavelet}")

        # Convolve rows with low and high filters
        low_rows = signal.convolve2d(current, low_filter[:, np.newaxis], mode='same', boundary='symm')[::2, :]
        high_rows = signal.convolve2d(current, high_filter[:, np.newaxis], mode='same', boundary='symm')[::2, :]

        # Convolve columns with low and high filters
        low_low = signal.convolve2d(low_rows, low_filter[np.newaxis, :], mode='same', boundary='symm')[:, ::2]
        low_high = signal.convolve2d(low_rows, high_filter[np.newaxis, :], mode='same', boundary='symm')[:, ::2]
        high_low = signal.convolve2d(high_rows, low_filter[np.newaxis, :], mode='same', boundary='symm')[:, ::2]
        high_high = signal.convolve2d(high_rows, high_filter[np.newaxis, :], mode='same', boundary='symm')[:, ::2]

        coeffs.append((low_low, low_high, high_low, high_high))
        current = low_low  # Proceed to next level

    return coeffs


def idwt2d_patch(coeffs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], wavelet: str = 'db1') -> np.ndarray:
    """
    Perform a multi-level 2D Inverse Discrete Wavelet Transform to reconstruct a single patch.

    Args:
        coeffs (List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]): List of coefficients at each level.
        wavelet (str, optional): Type of wavelet to use. Defaults to 'db1'.

    Returns:
        np.ndarray: Reconstructed 2D array representing the patch.
    """
    if wavelet == 'db1':
        low_filter = np.array([1 / math.sqrt(2), 1 / math.sqrt(2)])
        high_filter = np.array([1 / math.sqrt(2), -1 / math.sqrt(2)])
    else:
        raise ValueError(f"Unsupported wavelet type: {wavelet}")

    for level_coeffs in reversed(coeffs):
        low_low, low_high, high_low, high_high = level_coeffs

        # Upsample columns
        low_low_up = np.zeros((low_low.shape[0], low_low.shape[1] * 2))
        low_low_up[:, ::2] = low_low
        low_high_up = np.zeros((low_high.shape[0], low_high.shape[1] * 2))
        low_high_up[:, ::2] = low_high
        high_low_up = np.zeros((high_low.shape[0], high_low.shape[1] * 2))
        high_low_up[:, ::2] = high_low
        high_high_up = np.zeros((high_high.shape[0], high_high.shape[1] * 2))
        high_high_up[:, ::2] = high_high

        # Convolve columns with synthesis filters
        low_rows = signal.convolve2d(low_low_up, low_filter[:, np.newaxis], mode='same', boundary='symm') + \
                   signal.convolve2d(low_high_up, high_filter[:, np.newaxis], mode='same', boundary='symm')
        high_rows = signal.convolve2d(high_low_up, low_filter[:, np.newaxis], mode='same', boundary='symm') + \
                    signal.convolve2d(high_high_up, high_filter[:, np.newaxis], mode='same', boundary='symm')

        # Upsample rows
        low_rows_up = np.zeros((low_rows.shape[0] * 2, low_rows.shape[1]))
        low_rows_up[::2, :] = low_rows
        high_rows_up = np.zeros((high_rows.shape[0] * 2, high_rows.shape[1]))
        high_rows_up[::2, :] = high_rows

        # Convolve rows with synthesis filters and sum to reconstruct
        current = signal.convolve2d(low_rows_up, low_filter[np.newaxis, :], mode='same', boundary='symm') + \
                  signal.convolve2d(high_rows_up, high_filter[np.newaxis, :], mode='same', boundary='symm')

    return current


def wavelet_patch_fusion(
    tensor: torch.Tensor,
    patch_size: Tuple[int, int],
    wavelet: str = 'db1',
    level: int = 2,
    fusion_ratio: float = 0.05,
    random_seed: int = None
) -> torch.Tensor:
    """
    Applies Wavelet Patch Fusion on the input tensor without using `pywt`.

    Args:
        tensor (torch.Tensor): Input tensor of shape (B, C, H, W).
        patch_size (Tuple[int, int]): Size of each patch (patch_height, patch_width).
        wavelet (str, optional): Wavelet type to use for decomposition. Defaults to 'db1'.
        level (int, optional): Number of decomposition levels. Defaults to 2.
        fusion_ratio (float, optional): Ratio of patches to fuse. Defaults to 0.05 (5%).
        random_seed (int, optional): Seed for random number generator for reproducibility. Defaults to None.

    Returns:
        torch.Tensor: Tensor after applying wavelet patch fusion with shape (B, C, H, W).

    Raises:
        ValueError: If input tensor dimensions are incompatible with patch size.
    """
    try:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input tensor must be a torch.Tensor.")

        if tensor.dim() != 4:
            raise ValueError("Input tensor must have 4 dimensions (B, C, H, W).")

        batch_size, channels, height, width = tensor.shape
        patch_height, patch_width = patch_size

        if height % patch_height != 0 or width % patch_width != 0:
            raise ValueError(
                "Height and Width of the tensor must be divisible by the patch size."
            )

        # Calculate number of patches
        n_patches_h = height // patch_height
        n_patches_w = width // patch_width
        total_patches = batch_size * channels * n_patches_h * n_patches_w

        if total_patches == 0:
            raise ValueError("Total number of patches is zero. Check patch size and tensor dimensions.")

        # Set random seed for reproducibility if provided
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)  # Also set NumPy's seed for consistency

        # Determine number of patches to fuse
        num_fuse = max(1, math.ceil(total_patches * fusion_ratio))

        # Extract patches
        patches = tensor.unfold(2, patch_height, patch_height).unfold(3, patch_width, patch_width)
        # patches shape: (B, C, n_patches_h, n_patches_w, patch_height, patch_width)
        patches = patches.contiguous().view(-1, patch_height, patch_width)  # Shape: (B*C*n_patches, patch_h, patch_w)

        # Convert patches to numpy for processing
        patches_np = patches.cpu().numpy()

        # Randomly select patches to fuse
        if num_fuse > patches.size(0):
            raise ValueError("Number of patches to fuse exceeds total number of patches.")
        fuse_indices = random.sample(range(patches.size(0)), num_fuse)

        # Perform wavelet decomposition on selected patches
        coeffs_list = []
        for idx in fuse_indices:
            patch = patches_np[idx]
            coeffs = dwt2d_patch(patch, wavelet=wavelet, level=level)
            coeffs_list.append(coeffs)

        if not coeffs_list:
            raise RuntimeError("No coefficients were generated for fusion.")

        # Initialize fused coefficients with zeros
        fused_coeffs = []
        for level_idx in range(level):
            # Initialize accumulators for each coefficient type
            fused_LL = None
            fused_LH = None
            fused_HL = None
            fused_HH = None

            for coeff in coeffs_list:
                LL, LH, HL, HH = coeff[level_idx]
                if fused_LL is None:
                    fused_LL = np.zeros_like(LL)
                    fused_LH = np.zeros_like(LH)
                    fused_HL = np.zeros_like(HL)
                    fused_HH = np.zeros_like(HH)
                fused_LL += LL
                fused_LH += LH
                fused_HL += HL
                fused_HH += HH

            # Average the coefficients
            fused_LL /= num_fuse
            fused_LH /= num_fuse
            fused_HL /= num_fuse
            fused_HH /= num_fuse

            fused_coeffs.append((fused_LL, fused_LH, fused_HL, fused_HH))

        # Reconstruct fused patches
        fused_patches = []
        for _ in fuse_indices:
            reconstructed = idwt2d_patch(fused_coeffs, wavelet=wavelet)
            # Ensure the reconstructed patch has the exact patch size
            reconstructed = reconstructed[:patch_height, :patch_width]
            # Handle any NaN or Inf values
            reconstructed = np.nan_to_num(reconstructed, nan=0.0, posinf=0.0, neginf=0.0)
            fused_patches.append(reconstructed)

        if not fused_patches:
            raise RuntimeError("No patches were fused.")

        # Convert fused patches to a single NumPy array
        fused_patches_np = np.stack(fused_patches, axis=0)

        # Convert NumPy array to Torch tensor efficiently
        fused_patches_tensor = torch.from_numpy(fused_patches_np).type(tensor.dtype).to(tensor.device)

        # Replace original patches with fused patches
        patches[fuse_indices] = fused_patches_tensor

        # Reconstruct the tensor from patches
        patches = patches.view(batch_size, channels, n_patches_h, n_patches_w, patch_height, patch_width)
        # Permute to (B, C, n_patches_h, patch_height, n_patches_w, patch_width)
        patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
        # Reshape to (B, C, H, W)
        fused_tensor = patches.view(batch_size, channels, height, width)

        return fused_tensor

    except Exception as e:
        raise RuntimeError(f"Wavelet Patch Fusion failed: {e}") from e
