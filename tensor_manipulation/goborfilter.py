import math
from typing import Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor


def generate_gabor_kernel(
    kernel_size: int,
    sigma: float,
    theta: float,
    lambda_: float,
    gamma: float,
    psi: float = 0,
) -> Tensor:
    """
    Generates a 2D Gabor kernel.

    Args:
        kernel_size (int): Size of the Gabor kernel (must be odd).
        sigma (float): Standard deviation of the Gaussian envelope.
        theta (float): Orientation of the Gabor filter in radians.
        lambda_ (float): Wavelength of the sinusoidal factor.
        gamma (float): Spatial aspect ratio.
        psi (float, optional): Phase offset. Defaults to 0.

    Returns:
        Tensor: Gabor kernel of shape (1, 1, kernel_size, kernel_size).
    """
    if kernel_size % 2 == 0 or kernel_size <= 0:
        raise ValueError("kernel_size must be a positive odd integer.")

    half_size = kernel_size // 2
    y, x = torch.meshgrid(
        torch.arange(-half_size, half_size + 1),
        torch.arange(-half_size, half_size + 1),
    )
    x = x.float()
    y = y.float()

    # Rotation
    x_theta = x * math.cos(theta) + y * math.sin(theta)
    y_theta = -x * math.sin(theta) + y * math.cos(theta)

    # Gabor kernel formula
    gb = torch.exp(
        -0.5 * (x_theta ** 2 + (gamma ** 2) * y_theta ** 2) / (sigma ** 2)
    ) * torch.cos(2 * math.pi * x_theta / lambda_ + psi)

    # Normalize the kernel
    gb -= gb.mean()
    norm = torch.sqrt(torch.sum(gb ** 2))
    gb /= norm

    return gb.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, K, K)


def create_gabor_filters(
    kernel_size: int = 31,
    sigmas: List[float] = [4.0, 8.0, 16.0],
    thetas: List[float] = [0, math.pi / 4, math.pi / 2, 3 * math.pi / 4],
    lambd: float = 10.0,
    gamma: float = 0.5,
    psi: float = 0,
) -> Tensor:
    """
    Creates a set of Gabor filters with specified parameters.

    Args:
        kernel_size (int, optional): Size of each Gabor kernel. Defaults to 31.
        sigmas (List[float], optional): List of sigma values for different scales. Defaults to [4.0, 8.0, 16.0].
        thetas (List[float], optional): List of orientations in radians. Defaults to [0, π/4, π/2, 3π/4].
        lambd (float, optional): Wavelength of the sinusoidal factor. Defaults to 10.0.
        gamma (float, optional): Spatial aspect ratio. Defaults to 0.5.
        psi (float, optional): Phase offset. Defaults to 0.

    Returns:
        Tensor: Gabor filters of shape (num_filters, 1, K, K).
    """
    filters = []
    for sigma in sigmas:
        for theta in thetas:
            kernel = generate_gabor_kernel(
                kernel_size=kernel_size,
                sigma=sigma,
                theta=theta,
                lambda_=lambd,
                gamma=gamma,
                psi=psi,
            )
            filters.append(kernel)
    return torch.cat(filters, dim=0)  # Shape: (num_filters, 1, K, K)


def gabor_filter_patch_augmentation(
    input_tensor: Tensor,
    patch_size: Tuple[int, int],
    eps: float = 0.05,
) -> Tensor:
    """
    Applies Gabor filter patch augmentation to the input tensor.

    Args:
        input_tensor (Tensor): Input tensor of shape (B, C, H, W).
        patch_size (Tuple[int, int]): Size of each patch as (height, width).
        eps (float, optional): Epsilon value for numerical stability. Defaults to 0.05.

    Returns:
        Tensor: Augmented tensor with Gabor filtered channels, shape (B, C + C*num_filters, H, W).

    Raises:
        TypeError: If input types are incorrect.
        ValueError: If input dimensions are incorrect.
    """
    if not isinstance(input_tensor, Tensor):
        raise TypeError("input_tensor must be a torch.Tensor.")
    if not isinstance(patch_size, tuple) or len(patch_size) != 2:
        raise TypeError("patch_size must be a tuple of two integers.")
    if not isinstance(eps, float) and not isinstance(eps, int):
        raise TypeError("eps must be a float.")

    if input_tensor.dim() != 4:
        raise ValueError("input_tensor must have 4 dimensions (B, C, H, W).")

    batch_size, channels, height, width = input_tensor.shape
    patch_height, patch_width = patch_size

    if patch_height > height or patch_width > width:
        raise ValueError("patch_size must be smaller than or equal to input tensor dimensions.")

    # Define Gabor filter parameters
    kernel_size = 31
    sigmas = [4.0, 8.0, 16.0]
    thetas = [0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]
    lambd = 10.0
    gamma = 0.5
    psi = 0

    # Generate Gabor filters
    gabor_filters = create_gabor_filters(
        kernel_size=kernel_size,
        sigmas=sigmas,
        thetas=thetas,
        lambd=lambd,
        gamma=gamma,
        psi=psi,
    )  # Shape: (num_filters, 1, K, K)
    num_filters = gabor_filters.shape[0]

    # Adjust filters for input channels
    gabor_filters = gabor_filters.repeat(channels, 1, 1, 1)  # Shape: (C*num_filters, 1, K, K)

    # Apply padding to maintain spatial dimensions
    padding = kernel_size // 2

    # Perform convolution
    augmented = F.conv2d(
        input_tensor,
        weight=gabor_filters,
        bias=None,
        stride=1,
        padding=padding,
        groups=channels,
    )  # Shape: (B, C*num_filters, H, W)

    # Normalize augmented features with epsilon to avoid division by zero
    augmented = augmented / (torch.norm(augmented, dim=1, keepdim=True) + eps)

    # Concatenate the augmented features to the original tensor
    output = torch.cat([input_tensor, augmented], dim=1)  # Shape: (B, C + C*num_filters, H, W)

    return output

