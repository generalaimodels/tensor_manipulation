import torch
from torch import Tensor
from typing import Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TensorPatchMixer:
    """
    A class to shuffle and unshuffle patches of a tensor based on a given seed.
    """

    def __init__(self, patch_size: Tuple[int, int]) -> None:
        """
        Initializes the TensorPatchMixer with the specified patch size and seed.

        Args:
            patch_size (Tuple[int, int]): The size of each patch as (patch_height, patch_width).
           
        
        Raises:
            ValueError: If patch_size contains non-positive integers.
        """
        if (
            not isinstance(patch_size, tuple)
            or len(patch_size) != 2
            or not all(isinstance(x, int) and x > 0 for x in patch_size)
        ):
            logger.error("Invalid patch_size. It must be a tuple of two positive integers.")
            raise ValueError(
                "patch_size must be a tuple of two positive integers, e.g., (patch_height, patch_width)."
            )
        
        self.patch_height, self.patch_width = patch_size
        
        logger.debug(f"Initialized TensorPatchMixer with patch_size={patch_size} ")

    def _validate_tensor(self, tensor: Tensor) -> None:
        """
        Validates the input tensor's dimensions.

        Args:
            tensor (Tensor): The input tensor to validate.
        
        Raises:
            ValueError: If tensor does not have four dimensions.
        """
        if not isinstance(tensor, Tensor):
            logger.error("Input must be a torch.Tensor.")
            raise TypeError("Input must be a torch.Tensor.")
        if tensor.dim() != 4:
            logger.error("Tensor must have four dimensions (Batch, Channel, Height, Width).")
            raise ValueError("Tensor must have four dimensions (Batch, Channel, Height, Width).")
        logger.debug("Tensor validation passed.")

    def shuffle_tensor(self, tensor: Tensor,seed:int) -> Tensor:
        """
        Shuffles the patches of the input tensor based on the provided seed.

        Args:
            tensor (Tensor): The input tensor with shape (Batch, Channel, Height, Width).
        
        Returns:
            Tensor: The shuffled tensor with the same shape as input.
        
        Raises:
            ValueError: If tensor dimensions are not compatible with patch size.
        """
        self._validate_tensor(tensor)
        batch_size, channels, height, width = tensor.shape
        
        if height % self.patch_height != 0 or width % self.patch_width != 0:
            logger.error(
                "Height and Width must be divisible by patch dimensions."
            )
            raise ValueError(
                "Height and Width must be divisible by patch dimensions."
            )
        
        # Number of patches along height and width
        num_patches_h = height // self.patch_height
        num_patches_w = width // self.patch_width
        total_patches = num_patches_h * num_patches_w
        logger.debug(f"Patches - Height: {num_patches_h}, Width: {num_patches_w}, Total: {total_patches}")

        # Reshape tensor to extract patches
        patches = tensor.unfold(2, self.patch_height, self.patch_height) \
                        .unfold(3, self.patch_width, self.patch_width) \
                        .contiguous()
        logger.debug(f"Extracted patches shape: {patches.shape}")

        # Flatten patches
        patches = patches.view(batch_size, channels, total_patches, self.patch_height, self.patch_width)
        logger.debug(f"Flattened patches shape: {patches.shape}")

        # Shuffle patches
        generator = torch.Generator()
        generator.manual_seed(seed)
        permutation = torch.randperm(total_patches, generator=generator)
        logger.debug(f"Patch permutation: {permutation}")

        shuffled_patches = patches[:, :, permutation, :, :]
        logger.debug(f"Shuffled patches shape: {shuffled_patches.shape}")

        # Reshape back to original tensor shape
        shuffled_tensor = shuffled_patches.view(
            batch_size,
            channels,
            num_patches_h,
            num_patches_w,
            self.patch_height,
            self.patch_width
        )
        shuffled_tensor = shuffled_tensor.permute(0, 1, 2, 4, 3, 5).contiguous()
        shuffled_tensor = shuffled_tensor.view(batch_size, channels, height, width)
        logger.debug(f"Shuffled tensor shape: {shuffled_tensor.shape}")

        return shuffled_tensor

    def unshuffle_tensor(self, shuffled_tensor: Tensor,seed:int) -> Tensor:
        """
        Unshuffles the tensor to retrieve the original tensor based on the provided seed.

        Args:
            shuffled_tensor (Tensor): The shuffled tensor with shape (Batch, Channel, Height, Width).
        
        Returns:
            Tensor: The original tensor with the same shape as input.
        
        Raises:
            ValueError: If tensor dimensions are not compatible with patch size.
        """
        self._validate_tensor(shuffled_tensor)
        batch_size, channels, height, width = shuffled_tensor.shape
        
        if height % self.patch_height != 0 or width % self.patch_width != 0:
            logger.error(
                "Height and Width must be divisible by patch dimensions."
            )
            raise ValueError(
                "Height and Width must be divisible by patch dimensions."
            )
        
        # Number of patches along height and width
        num_patches_h = height // self.patch_height
        num_patches_w = width // self.patch_width
        total_patches = num_patches_h * num_patches_w  # Corrected Calculation
        logger.debug(f"Patches - Height: {num_patches_h}, Width: {num_patches_w}, Total: {total_patches}")

        # Reshape shuffled tensor to extract patches
        patches = shuffled_tensor.unfold(2, self.patch_height, self.patch_height) \
                                .unfold(3, self.patch_width, self.patch_width) \
                                .contiguous()
        logger.debug(f"Extracted shuffled patches shape: {patches.shape}")

        # Flatten patches
        patches = patches.view(batch_size, channels, total_patches, self.patch_height, self.patch_width)
        logger.debug(f"Flattened shuffled patches shape: {patches.shape}")

        # Generate the same permutation to invert
        generator = torch.Generator()
        generator.manual_seed(seed)
        permutation = torch.randperm(total_patches, generator=generator)
        logger.debug(f"Inverse patch permutation: {permutation}")

        # Calculate the inverse permutation indices
        inverse_permutation = torch.argsort(permutation)
        logger.debug(f"Inverse permutation: {inverse_permutation}")

        unshuffled_patches = patches[:, :, inverse_permutation, :, :]
        logger.debug(f"Unshuffled patches shape: {unshuffled_patches.shape}")

        # Reshape back to original tensor shape
        unshuffled_tensor = unshuffled_patches.view(
            batch_size,
            channels,
            num_patches_h,
            num_patches_w,
            self.patch_height,
            self.patch_width
        )
        unshuffled_tensor = unshuffled_tensor.permute(0, 1, 2, 4, 3, 5).contiguous()
        unshuffled_tensor = unshuffled_tensor.view(batch_size, channels, height, width)
        logger.debug(f"Unshuffled tensor shape: {unshuffled_tensor.shape}")

        return unshuffled_tensor

