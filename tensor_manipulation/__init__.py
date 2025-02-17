from .anisotropicdiffusion import anisotropic_diffusion_patch_filtering
from .autoregressive import autoregressive_patch_prediction
from .covariance import covariance_patch_whitening
from .crossdomainrandom import cross_domain_random_permutation
from .entropybased import entropy_based_patch_swap
from .directionalpatchgradient import directional_patch_gradient_swapping
from .fourierpatch import fourier_patch_mixing
from .frequencyaware import frequency_aware_patch_scaling
from .frequencyselective import frequency_selective_patch_masking
from .goborfilter import gabor_filter_patch_augmentation
from .kurtosisbasedpatch import kurtosis_based_patch_selection
from .laplacianpyramid import laplacian_pyramid_blending
from .localbinarypattern import apply_lbp_patch_encoding
from .mrfpatchstitching import mrf_patch_stitching
from .patchmixer import TensorPatchMixer
from .randomgaussiannoise import add_random_gaussian_noise_to_patches
from .stochasticpatch import stochastic_patch_replacement
from .temporalpatch import temporal_patch_mixing
from .texturebasedpatch import texture_based_patch_reorganization
from .waveletpatch import wavelet_patch_fusion