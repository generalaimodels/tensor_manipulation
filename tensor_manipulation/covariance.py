import torch
import torch.nn.functional as F
from typing import Tuple


def covariance_patch_whitening(
    tensor: torch.Tensor,
    patch_size: Tuple[int, int],
    epsilon: float = 1e-5
) -> torch.Tensor:
    """
    Perform ZCA whitening on patches of the input tensor to decorrelate pixel values
    while preserving spatial structure.

    Args:
        tensor (torch.Tensor): Input tensor with shape (batch_size, channels, height, width).
        patch_size (Tuple[int, int]): The size of each patch as (patch_height, patch_width).
        epsilon (float, optional): Small constant for numerical stability. Defaults to 1e-5.

    Returns:
        torch.Tensor: Whitened tensor with the same shape as the input (batch_size, channels, height, width).

    Raises:
        TypeError: If input types are incorrect.
        ValueError: If input tensor dimensions do not match expected dimensions.
    """
    # Input validation
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input 'tensor' must be a torch.Tensor.")
    
    if tensor.dim() != 4:
        raise ValueError(f"Input tensor must be 4-dimensional, got {tensor.dim()} dimensions.")
    
    if (
        not isinstance(patch_size, tuple) or
        len(patch_size) != 2 or
        not all(isinstance(dim, int) and dim > 0 for dim in patch_size)
    ):
        raise TypeError("Input 'patch_size' must be a tuple of two positive integers.")
    
    if not isinstance(epsilon, float) or epsilon <= 0:
        raise ValueError("Input 'epsilon' must be a positive float.")
    
    batch_size, channels, height, width = tensor.shape
    patch_height, patch_width = patch_size

    if height % patch_height != 0 or width % patch_width != 0:
        raise ValueError(
            f"Height ({height}) and width ({width}) must be divisible by patch size "
            f"({patch_height}, {patch_width})."
        )
    
    try:
        # Extract patches using unfold with non-overlapping stride
        patches = F.unfold(
            tensor,
            kernel_size=patch_size,
            stride=patch_size
        )  # Shape: (batch_size, channels * patch_height * patch_width, num_patches)
        
        num_patches = patches.shape[-1]
        
        # Reshape patches to (batch_size * num_patches, channels * patch_height * patch_width)
        patches = patches.permute(0, 2, 1).contiguous()
        patches = patches.view(-1, channels * patch_height * patch_width)
        
        # Compute the mean across all patches
        mean = patches.mean(dim=0, keepdim=True)  # Shape: (1, channels * patch_height * patch_width)
        
        # Center the patches
        patches_centered = patches - mean  # Shape: (batch_size * num_patches, channels * patch_height * patch_width)
        
        # Compute covariance matrix
        covariance = torch.matmul(patches_centered.T, patches_centered) / (patches_centered.shape[0] - 1)
        # Shape: (channels * patch_height * patch_width, channels * patch_height * patch_width)
        
        # Eigen-decomposition of covariance matrix
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        
        # Construct the ZCA whitening matrix
        # Add epsilon for numerical stability
        whitening_matrix = eigenvectors @ torch.diag(1.0 / torch.sqrt(eigenvalues + epsilon)) @ eigenvectors.T
        
        # Apply ZCA whitening
        patches_whitened = torch.matmul(patches_centered, whitening_matrix)
        
        # Reshape back to (batch_size, channels * patch_height * patch_width, num_patches)
        patches_whitened = patches_whitened.view(batch_size, num_patches, channels * patch_height * patch_width)
        patches_whitened = patches_whitened.permute(0, 2, 1).contiguous()
        
        # Reconstruct the tensor using fold
        tensor_whitened = F.fold(
            patches_whitened,
            output_size=(height, width),
            kernel_size=patch_size,
            stride=patch_size
        )  # Shape: (batch_size, channels, height, width)
        
        return tensor_whitened

    except RuntimeError as e:
        raise RuntimeError(f"An error occurred during ZCA whitening: {e}") from e