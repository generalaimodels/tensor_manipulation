import torch
import torch.nn.functional as F
import torch.fft
from typing import Tuple
import math


def fourier_patch_mixing(
    input_tensor: torch.Tensor,
    patch_size: Tuple[int, int],
    eps: float = 0.05
) -> torch.Tensor:
    """
    Applies Fourier Patch Mixing to the input tensor.
    
    Parameters:
        input_tensor (torch.Tensor): Input tensor of shape (B, C, H, W).
        patch_size (Tuple[int, int]): Size of each patch as (patch_height, patch_width).
        eps (float, optional): Percentage of patches to modify (default is 0.05 for 5%).
        
    Returns:
        torch.Tensor: Tensor after applying Fourier Patch Mixing with shape (B, C, H, W).
        
    Raises:
        ValueError: If input parameters are invalid.
        RuntimeError: If tensor operations fail.
    """
    try:
        # Validate input tensor dimensions
        if input_tensor.dim() != 4:
            raise ValueError(f"Input tensor must be 4-dimensional (B, C, H, W), got {input_tensor.dim()} dimensions.")

        batch_size, channels, height, width = input_tensor.shape
        patch_height, patch_width = patch_size

        # Validate patch size
        if patch_height <= 0 or patch_width <= 0:
            raise ValueError("Patch dimensions must be positive integers.")
        if height < patch_height or width < patch_width:
            raise ValueError("Patch size must be smaller than or equal to the tensor dimensions.")

        # Calculate number of patches along height and width
        num_patches_h = height // patch_height
        num_patches_w = width // patch_width
        total_patches = num_patches_h * num_patches_w

        if total_patches < 2:
            raise ValueError("Insufficient number of patches to perform magnitude swapping. "
                             "Consider reducing patch size or increasing tensor dimensions.")

        # Calculate number of patches to modify
        raw_num_modify = math.floor(eps * total_patches)
        num_modify = max(2, raw_num_modify)
        # Ensure num_modify is even
        if num_modify % 2 != 0:
            num_modify = num_modify - 1 if num_modify > 2 else num_modify + 1

        # Ensure num_modify does not exceed total_patches
        num_modify = min(num_modify, total_patches if total_patches % 2 == 0 else total_patches - 1)

        if num_modify < 2:
            raise ValueError("Not enough patches selected to perform magnitude swapping after adjustments.")

        # Extract patches using unfold
        patches = F.unfold(
            input_tensor, 
            kernel_size=patch_size, 
            stride=patch_size
        )  # Shape: (B, C * patch_height * patch_width, L)
        
        B, C_patch, L = patches.shape
        C = channels
        P_h, P_w = patch_size
        P = P_h * P_w

        if L != total_patches:
            raise ValueError(f"Expected {total_patches} patches, but got {L} patches.")

        # Reshape patches to (B, L, C, H_p, W_p)
        patches = patches.view(B, C, P_h, P_w, L)  # (B, C, H_p, W_p, L)
        patches = patches.permute(0, 4, 1, 2, 3).contiguous()  # (B, L, C, H_p, W_p)

        # Select random patch indices to modify
        rand_indices = torch.randperm(L)[:num_modify]

        # Apply Fourier Transform to selected patches
        selected_patches = patches[:, rand_indices, :, :, :]  # (B, num_modify, C, H_p, W_p)
        selected_patches_complex = torch.fft.fft2(selected_patches, dim=(-2, -1))
        
        # Ensure even number for swapping
        if num_modify < 2:
            raise ValueError("Not enough patches selected to perform magnitude swapping.")

        # Reshape selected patches into pairs for swapping
        patch_pairs = selected_patches_complex.view(B, num_modify // 2, 2, C, P_h, P_w)
        magnitudes = torch.abs(patch_pairs)
        phases = torch.angle(patch_pairs)

        # Swap magnitudes between consecutive patches in each pair
        magnitudes_swapped = magnitudes[:, :, [1, 0], :, :, :]

        # Reconstruct complex patches with swapped magnitudes and original phases
        patches_swapped = magnitudes_swapped * torch.exp(1j * phases)

        # Reshape back to original selected patches shape
        patches_swapped = patches_swapped.view(B, num_modify, C, P_h, P_w)

        # Apply inverse Fourier Transform
        patches_swapped_spatial = torch.fft.ifft2(patches_swapped, dim=(-2, -1)).real

        # Update the selected patches with modified patches
        patches[:, rand_indices, :, :, :] = patches_swapped_spatial

        # Reshape patches back to (B, C * P_h * P_w, L)
        patches = patches.permute(0, 2, 3, 4, 1).contiguous()  # (B, C, H_p, W_p, L)
        patches = patches.view(B, C * P_h * P_w, L)

        # Reconstruct the tensor using fold
        output_tensor = F.fold(
            patches, 
            output_size=(num_patches_h * patch_height, num_patches_w * patch_width),
            kernel_size=patch_size, 
            stride=patch_size
        )  # (B, C * P_h * P_w, H, W)

        # Reshape to (B, C, H, W)
        output_tensor = output_tensor.view(B, channels, height, width)

        return output_tensor

    except Exception as e:
        raise RuntimeError(f"Fourier Patch Mixing failed: {str(e)}")

