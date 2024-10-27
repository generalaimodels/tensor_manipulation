import torch
import torch.nn.functional as F
from typing import Tuple


def frequency_selective_patch_masking(
    input_tensor: torch.Tensor,
    patch_size: Tuple[int, int],
    eps: float = 0.05
) -> torch.Tensor:
    """
    Perform Frequency-Selective Patch Masking on a 4D tensor.

    This function divides the input tensor into non-overlapping patches, applies a Fourier
    transform to each patch, masks out the lowest `eps` percentage of frequency components
    based on their magnitude, and reconstructs the tensor via an inverse Fourier transform.

    Args:
        input_tensor (torch.Tensor): The input tensor with shape
            (batch_size, channels, height, width).
        patch_size (Tuple[int, int]): The size of each patch as (patch_height, patch_width).
        eps (float, optional): The percentage of low-magnitude frequency components to mask out.
            Must be between 0 and 1. Defaults to 0.05 (5%).

    Returns:
        torch.Tensor: The masked tensor with the same shape as `input_tensor`.

    Raises:
        ValueError: If input dimensions are invalid or parameters are out of bounds.
        TypeError: If input types are incorrect.
    """
    # Input validation
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError(
            f"input_tensor must be a torch.Tensor, but got {type(input_tensor)}."
        )
    if input_tensor.dim() != 4:
        raise ValueError(
            f"input_tensor must be a 4D tensor, but got {input_tensor.dim()}D."
        )
    if not isinstance(patch_size, tuple) or len(patch_size) != 2:
        raise TypeError(
            "patch_size must be a tuple of two integers, e.g., (16, 16)."
        )
    if not all(isinstance(dim, int) and dim > 0 for dim in patch_size):
        raise ValueError(
            "patch_size dimensions must be positive integers."
        )
    if not isinstance(eps, float) or not (0.0 < eps < 1.0):
        raise ValueError(
            "eps must be a float between 0 and 1 (exclusive)."
        )

    batch_size, channels, height, width = input_tensor.shape
    patch_height, patch_width = patch_size

    # Check if height and width are divisible by patch dimensions
    if height % patch_height != 0 or width % patch_width != 0:
        raise ValueError(
            "Height and width of input_tensor must be divisible by patch dimensions."
        )

    num_patches_h = height // patch_height
    num_patches_w = width // patch_width

    # Reshape to (batch_size, channels, num_patches_h, patch_height, num_patches_w, patch_width)
    patches = input_tensor.view(
        batch_size,
        channels,
        num_patches_h,
        patch_height,
        num_patches_w,
        patch_width
    )

    # Permute to (batch_size, channels, num_patches_h, num_patches_w, patch_height, patch_width)
    patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()

    # Flatten batch and channel dimensions for efficient processing
    patches = patches.view(-1, patch_height, patch_width)  # Shape: (batch_size * channels * num_patches_h * num_patches_w, H, W)

    # Perform FFT on the patches
    patches_fft = torch.fft.fft2(patches)
    patches_fft_shifted = torch.fft.fftshift(patches_fft, dim=(-2, -1))  # Shift zero freq to center

    # Create frequency mask
    # Calculate the number of frequencies to mask based on eps
    total_freq = patch_height * patch_width
    num_mask = int(total_freq * eps)

    if num_mask == 0:
        raise ValueError(
            f"eps={eps} is too small. It results in masking zero frequency components."
        )
    if num_mask >= total_freq:
        raise ValueError(
            f"eps={eps} is too large. It results in masking all frequency components."
        )

    # Flatten the frequency magnitudes to find low-magnitude frequencies
    freq_magnitude = patches_fft_shifted.abs().flatten(-2, -1)  # Shape: (num_patches, H*W)

    # Get the threshold magnitude for masking
    # torch.kthvalue finds the kth smallest value, so for lowest 'num_mask' magnitudes
    threshold, _ = torch.kthvalue(freq_magnitude, num_mask, dim=-1, keepdim=True)  # Shape: (num_patches, 1)

    # Create mask: 1 where magnitude >= threshold, else 0
    mask = (freq_magnitude >= threshold).float()  # Shape: (num_patches, H*W)

    # Reshape mask to (num_patches, H, W)
    mask = mask.view(-1, patch_height, patch_width)  # Shape: (num_patches, H, W)

    # Apply the mask
    patches_fft_shifted_masked = patches_fft_shifted * mask  # Shape: (num_patches, H, W)

    # Inverse FFT shift
    patches_fft_unshifted = torch.fft.ifftshift(patches_fft_shifted_masked, dim=(-2, -1))

    # Inverse FFT to get masked patches
    patches_masked = torch.fft.ifft2(patches_fft_unshifted).real  # Take the real part

    # Reshape back to original tensor shape
    patches_masked = patches_masked.view(
        batch_size,
        channels,
        num_patches_h,
        num_patches_w,
        patch_height,
        patch_width
    )

    # Permute back to (batch_size, channels, height, width)
    patches_masked = patches_masked.permute(0, 1, 2, 4, 3, 5).contiguous()
    output_tensor = patches_masked.view(batch_size, channels, height, width)

    return output_tensor

