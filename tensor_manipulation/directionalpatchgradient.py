import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


class DirectionalPatchGradientSwapper:
    """
    A class to perform directional patch gradient swapping on input tensors.
    """

    def __init__(self, patch_size: Tuple[int, int], epsilon: float = 0.05) -> None:
        """
        Initialize the gradient swapper.

        Args:
            patch_size (Tuple[int, int]): Size of each patch (height, width).
            epsilon (float): Tolerance percentage for gradient constraints.
        """
        if not isinstance(patch_size, tuple) or len(patch_size) != 2:
            raise ValueError("patch_size must be a tuple of two integers (height, width).")
        if not (0 < epsilon < 1):
            raise ValueError("epsilon must be a float between 0 and 1.")

        self.patch_height, self.patch_width = patch_size
        self.epsilon = epsilon
        self.sobel_kernel_x = self._create_sobel_kernel(axis='x', device='cpu')
        self.sobel_kernel_y = self._create_sobel_kernel(axis='y', device='cpu')

    @staticmethod
    def _create_sobel_kernel(axis: str, device: torch.device) -> Tensor:
        """
        Create a Sobel kernel for gradient computation.

        Args:
            axis (str): 'x' or 'y' axis for the Sobel filter.
            device (torch.device): Device to create the kernel on.

        Returns:
            Tensor: Sobel kernel of shape (1, 1, 3, 3).
        """
        if axis == 'x':
            kernel = torch.tensor(
                [[-1, 0, 1],
                 [-2, 0, 2],
                 [-1, 0, 1]], dtype=torch.float32
            )
        elif axis == 'y':
            kernel = torch.tensor(
                [[-1, -2, -1],
                 [0, 0, 0],
                 [1, 2, 1]], dtype=torch.float32
            )
        else:
            raise ValueError("Axis must be either 'x' or 'y'.")

        return kernel.view(1, 1, 3, 3).to(device)

    def _extract_patches(self, x: Tensor) -> Tuple[Tensor, int, int]:
        """
        Extract non-overlapping patches from the input tensor.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tuple[Tensor, int, int]: 
                - Extracted patches of shape (B, C, N, patch_height, patch_width).
                - Number of patches along the height.
                - Number of patches along the width.
        """
        B, C, H, W = x.shape
        if H % self.patch_height != 0 or W % self.patch_width != 0:
            raise ValueError("Height and Width must be divisible by patch dimensions.")

        n_patches_h = H // self.patch_height
        n_patches_w = W // self.patch_width
        N = n_patches_h * n_patches_w

        patches = x.unfold(2, self.patch_height, self.patch_height)\
                   .unfold(3, self.patch_width, self.patch_width)
        patches = patches.contiguous().view(B, C, N, self.patch_height, self.patch_width)
        return patches, n_patches_h, n_patches_w

    def _compute_gradients(self, patches: Tensor, device: torch.device) -> Tuple[Tensor, Tensor]:
        """
        Compute directional gradients for each patch.

        Args:
            patches (Tensor): Patches tensor of shape (B, C, N, H_p, W_p).
            device (torch.device): Device for computation.

        Returns:
            Tuple[Tensor, Tensor]: Gradient maps along x and y axes.
        """
        B, C, N, H_p, W_p = patches.shape
        patches = patches.view(B * C * N, 1, H_p, W_p)

        grad_x = F.conv2d(patches, self.sobel_kernel_x.to(device), padding=1)
        grad_y = F.conv2d(patches, self.sobel_kernel_y.to(device), padding=1)

        grad_x = grad_x.view(B, C, N, H_p, W_p)
        grad_y = grad_y.view(B, C, N, H_p, W_p)

        return grad_x, grad_y

    def _swap_gradients(self, grad_x: Tensor, grad_y: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Swap gradient maps between patches.

        Args:
            grad_x (Tensor): Gradient maps along x-axis.
            grad_y (Tensor): Gradient maps along y-axis.

        Returns:
            Tuple[Tensor, Tensor]: Swapped gradient maps.
        """
        B, C, N, _, _ = grad_x.shape
        # Generate a random permutation for swapping
        indices = torch.randperm(N)
        indices = indices.to(grad_x.device)
        swapped_grad_x = grad_x[:, :, indices, :, :]
        swapped_grad_y = grad_y[:, :, indices, :, :]
        return swapped_grad_x, swapped_grad_y

    def _reconstruct_patches(
        self,
        patches: Tensor,
        swapped_grad_x: Tensor,
        swapped_grad_y: Tensor,
        device: torch.device
    ) -> Tensor:
        """
        Reconstruct patches using swapped gradients.

        Args:
            patches (Tensor): Original patches.
            swapped_grad_x (Tensor): Swapped gradient maps along x-axis.
            swapped_grad_y (Tensor): Swapped gradient maps along y-axis.
            device (torch.device): Device for computation.

        Returns:
            Tensor: Reconstructed patches.
        """
        # Placeholder for reconstruction logic.
        # This can be implemented using gradient-based image reconstruction techniques.
        # For simplicity, we'll blend the original patch with gradient information.

        # Compute magnitude of new gradients
        grad_mag = torch.sqrt(swapped_grad_x ** 2 + swapped_grad_y ** 2 + 1e-8)
        grad_mag = grad_mag.mean(dim=1, keepdim=True)  # Average over channels

        # Apply epsilon constraint
        grad_threshold = self.epsilon * grad_mag.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        grad_mag = torch.clamp(grad_mag, max=grad_threshold)

        # Blend original patches with gradient magnitudes
        reconstructed = patches + grad_mag
        return reconstructed

    def swap_gradients(self, x: Tensor) -> Tensor:
        """
        Perform directional patch gradient swapping on the input tensor.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Output tensor of shape (B, C, H, W).
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        if x.dim() != 4:
            raise ValueError("Input tensor must have 4 dimensions (B, C, H, W).")

        device = x.device
        patches, n_patches_h, n_patches_w = self._extract_patches(x)
        grad_x, grad_y = self._compute_gradients(patches, device)
        swapped_grad_x, swapped_grad_y = self._swap_gradients(grad_x, grad_y)
        reconstructed_patches = self._reconstruct_patches(patches, swapped_grad_x, swapped_grad_y, device)

        B, C, N, H_p, W_p = reconstructed_patches.shape
        H_in, W_in = x.shape[2], x.shape[3]

        if N != n_patches_h * n_patches_w:
            raise ValueError("Number of patches does not match the expected grid.")

        # Reshape patches to (B, C, n_patches_h, n_patches_w, H_p, W_p)
        reconstructed_patches = reconstructed_patches.view(B, C, n_patches_h, n_patches_w, H_p, W_p)
        # Permute to (B, C, n_patches_h, H_p, n_patches_w, W_p)
        reconstructed_patches = reconstructed_patches.permute(0, 1, 2, 4, 3, 5).contiguous()
        # Reshape to (B, C, n_patches_h * H_p, n_patches_w * W_p)
        reconstructed = reconstructed_patches.view(B, C, n_patches_h * H_p, n_patches_w * W_p)

        return reconstructed


def directional_patch_gradient_swapping(
    tensor: Tensor,
    patch_size: Tuple[int, int],
    epsilon: float = 0.05
) -> Tensor:
    """
    Perform directional patch gradient swapping on the input tensor.

    Args:
        tensor (Tensor): Input tensor of shape (B, C, H, W).
        patch_size (Tuple[int, int]): Size of each patch (height, width).
        epsilon (float, optional): Tolerance percentage for gradient constraints. Defaults to 0.05.

    Returns:
        Tensor: Output tensor of shape (B, C, H, W).
    """
    swapper = DirectionalPatchGradientSwapper(patch_size=patch_size, epsilon=epsilon)
    return swapper.swap_gradients(tensor)


