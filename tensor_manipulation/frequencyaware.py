import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
import math


def dct(x: Tensor, norm: str = 'ortho') -> Tensor:
    """
    Discrete Cosine Transform (DCT) Type-II for real input.

    Args:
        x (Tensor): Input tensor.
        norm (str, optional): Normalization mode. Defaults to 'ortho'.

    Returns:
        Tensor: DCT transformed tensor.
    """
    N = x.shape[-1]
    x = torch.cat([x, x.flip(dims=[-1])], dim=-1)
    X = torch.fft.fft(x, dim=-1)
    real = X.real[..., :N]
    if norm == 'ortho':
        factor = math.sqrt(2.0 / N)
        real[..., 0] = real[..., 0] / math.sqrt(2)
        return real * factor
    else:
        return real


def idct(X: Tensor, norm: str = 'ortho') -> Tensor:
    """
    Inverse Discrete Cosine Transform (DCT) Type-II for real input.

    Args:
        X (Tensor): DCT transformed tensor.
        norm (str, optional): Normalization mode. Defaults to 'ortho'.

    Returns:
        Tensor: Inverse DCT transformed tensor.
    """
    N = X.shape[-1]
    if norm == 'ortho':
        factor = math.sqrt(2.0 / N)
        X = X / factor
        X[..., 0] = X[..., 0] * math.sqrt(2)
    # Create a symmetric tensor for inverse FFT
    X_sym = torch.cat([X, X.flip(dims=[-1])], dim=-1)
    x = torch.fft.ifft(torch.tensor(X_sym), dim=-1).real[..., :N]
    return x


def frequency_aware_patch_scaling(
    tensor: Tensor,
    patch_size: Tuple[int, int],
    eps: float = 0.05
) -> Tensor:
    """
    Applies frequency-aware scaling to patches of the input tensor using Discrete Cosine Transform (DCT).
    High frequencies are attenuated while mid-level frequencies are boosted.

    Args:
        tensor (Tensor): Input tensor of shape (batch_size, channels, height, width).
        patch_size (Tuple[int, int]): Size of the patches (patch_height, patch_width).
        eps (float, optional): Epsilon value to prevent division by zero. Defaults to 0.05.

    Returns:
        Tensor: Tensor after frequency-aware patch scaling with shape (batch_size, channels, height, width).

    Raises:
        ValueError: If input tensor does not have 4 dimensions.
        ValueError: If patch dimensions are larger than the input tensor's spatial dimensions.
    """
    if not isinstance(tensor, Tensor):
        raise TypeError("Input must be a PyTorch Tensor.")

    if tensor.dim() != 4:
        raise ValueError(f"Expected tensor with 4 dimensions, got {tensor.dim()}.")

    batch_size, channels, height, width = tensor.shape
    patch_height, patch_width = patch_size

    if patch_height > height or patch_width > width:
        raise ValueError("Patch size must be smaller than or equal to tensor's spatial dimensions.")

    try:
        # Calculate the number of patches along height and width
        num_patches_h = height // patch_height
        num_patches_w = width // patch_width

        # Trim the tensor to make it divisible by the patch size
        trimmed_height = num_patches_h * patch_height
        trimmed_width = num_patches_w * patch_width
        tensor = tensor[:, :, :trimmed_height, :trimmed_width]

        # Reshape tensor to (batch_size, channels, num_patches_h, patch_height, num_patches_w, patch_width)
        tensor_patches = tensor.reshape(
            batch_size,
            channels,
            num_patches_h,
            patch_height,
            num_patches_w,
            patch_width
        ).permute(0, 1, 2, 4, 3, 5).contiguous()

        # Merge batch and channels for processing
        tensor_patches = tensor_patches.view(-1, 1, patch_height, patch_width)

        # Apply DCT (Type-II) on the patches manually
        patches_dct = dct(dct(tensor_patches, norm='ortho'), norm='ortho')

        # Create frequency scaling mask
        freq_mask = create_frequency_mask(patch_height, patch_width, device=tensor.device)

        # Apply scaling: attenuate high frequencies, boost mid frequencies
        scaled_dct = patches_dct * freq_mask

        # Apply inverse DCT (Type-III) manually to get back to spatial domain
        patches_idct = idct(idct(scaled_dct, norm='ortho'), norm='ortho')

        # Reshape back to original tensor shape
        patches_idct = patches_idct.view(
            batch_size,
            channels,
            num_patches_h,
            num_patches_w,
            patch_height,
            patch_width
        ).permute(0, 1, 2, 4, 3, 5).contiguous()

        # Combine patches back to the original tensor shape
        output = patches_idct.view(batch_size, channels, trimmed_height, trimmed_width)

        return output
    except Exception as e:
        raise RuntimeError(f"An error occurred during frequency-aware patch scaling: {e}")


def create_frequency_mask(patch_height: int, patch_width: int, device: torch.device = torch.device('cpu')) -> Tensor:
    """
    Creates a frequency scaling mask that attenuates high frequencies and boosts mid-level frequencies.

    Args:
        patch_height (int): Height of the patch.
        patch_width (int): Width of the patch.
        device (torch.device, optional): Device for the mask. Defaults to CPU.

    Returns:
        Tensor: Frequency scaling mask of shape (1, 1, patch_height, patch_width).
    """
    y = torch.linspace(0, 1, steps=patch_height, device=device).unsqueeze(1).repeat(1, patch_width)
    x = torch.linspace(0, 1, steps=patch_width, device=device).unsqueeze(0).repeat(patch_height, 1)
    distance = torch.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
    mask = 1 - (distance / distance.max())  # Invert distance: center has high values
    # Attenuate high frequencies and boost mid frequencies
    mask = mask ** 2  # Quadratic scaling for smoother attenuation
    mask = mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    return mask

