import torch
import torch.nn.functional as F
from typing import Tuple

def apply_lbp_patch_encoding(
    input_tensor: torch.Tensor, 
    eps: float = 0.05
) -> torch.Tensor:
    """
    Apply Local Binary Pattern (LBP) encoding to each patch of the input tensor.
    
    This function enhances texture-based features by encoding local intensity 
    differences into binary patterns and appending them as additional feature channels.
    
    Args:
        input_tensor (torch.Tensor): 
            A tensor of shape (batch_size, channels, height, width).
        eps (float, optional): 
            A small epsilon value for numerical stability. Defaults to 0.05.
    
    Returns:
        torch.Tensor: 
            A tensor of shape (batch_size, channels + 1, height, width) 
            with LBP encoding as an additional channel.
    
    Raises:
        TypeError: 
            If input_tensor is not a torch.Tensor.
        ValueError: 
            If input_tensor does not have 4 dimensions.
        ValueError: 
            If eps is not between 0 and 1.
    """
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError("input_tensor must be a torch.Tensor.")
    
    if input_tensor.dim() != 4:
        raise ValueError("input_tensor must have 4 dimensions (batch_size, channels, height, width).")
    
    if not (0 < eps < 1):
        raise ValueError("eps must be between 0 and 1.")
    
    batch_size, channels, height, width = input_tensor.shape
    
    # Define LBP weights for 8 neighbors
    lbp_weights = torch.tensor(
        [1, 2, 4, 8, 16, 32, 64, 128],
        dtype=input_tensor.dtype,
        device=input_tensor.device
    )
    
    # Define padding for 3x3 LBP
    padding = (1, 1, 1, 1)
    
    # Pad the input tensor to handle borders
    padded_tensor = F.pad(input_tensor, padding, mode='replicate')
    
    # Initialize LBP feature tensor
    lbp_feature = torch.zeros((batch_size, channels, height, width), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Define neighbor shifts
    shifts = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, 1), (1, 1), (1, 0),
        (1, -1), (0, -1)
    ]
    
    for idx, (dy, dx) in enumerate(shifts):
        # Shift the padded tensor
        shifted = padded_tensor[:, :, 1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
        # Compare shifted tensor with center
        binary = (shifted >= input_tensor).float()
        # Accumulate weighted binary patterns
        lbp_feature += binary * lbp_weights[idx]
    
    # Normalize the LBP feature with epsilon to avoid division by zero
    lbp_feature = lbp_feature / (lbp_weights.sum() + eps)
    
    # Append the LBP feature as an additional channel
    output_tensor = torch.cat((input_tensor, lbp_feature), dim=1)
    
    return output_tensor

# Example usage
if __name__ == "__main__":
    # Create a random tensor with shape (batch_size, channels, height, width)
    batch_size, channels, height, width = 8, 3, 64, 64
    input_tensor = torch.rand(batch_size, channels, height, width)
    
    # Apply LBP patch encoding
    output_tensor = apply_lbp_patch_encoding(input_tensor, eps=0.05)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")