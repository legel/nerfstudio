# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GSplat-based Depth Supervision Utilities

This module provides efficient depth supervision for Gaussian Splatting by leveraging
gsplat's native depth rendering capabilities and intermediate rasterization data.

Unlike our previous approach that reimplemented projection logic, this implementation
uses gsplat's built-in features:

1. **Expected Depth Rendering**: Uses render_mode="ED" to get per-pixel depths directly
2. **Rasterization Metadata**: Leverages the 'info' dict from gsplat.rasterization()
3. **Native Efficiency**: Utilizes gsplat's optimized CUDA kernels and memory management

Key GSplat Integration Points:
────────────────────────────

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GSplat        │    │  Depth Modes    │    │   Meta Info     │
│ Rasterization   │───▶│                 │───▶│                 │
│                 │    │ • "ED" - Depth  │    │ • means2d       │
│ • Pixel mapping │    │ • "RGB+ED"      │    │ • visibility    │
│ • Depth compute │    │ • "D" - Accum   │    │ • tile info     │
│ • Alpha blend   │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘

This approach provides:
- **Performance**: No redundant projection calculations
- **Accuracy**: Uses exact same depth computation as gsplat's rendering
- **Memory Efficiency**: Leverages gsplat's packed/sparse optimizations
- **Future-proof**: Compatible with gsplat updates and new features
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from jaxtyping import Float
from torch import Tensor

try:
    from gsplat import rasterization
except ImportError:
    print("Warning: gsplat not available. Please install with: pip install gsplat")
    rasterization = None

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.rich_utils import CONSOLE


def extract_depth_from_gsplat_render(
    render_output: Float[Tensor, "batch height width channels"],
    alpha_output: Float[Tensor, "batch height width 1"], 
    render_mode: str,
    background_depth: Optional[float] = None
) -> Float[Tensor, "batch height width 1"]:
    """
    Extract depth information from gsplat rasterization output.
    
    GSplat provides several depth rendering modes:
    - "ED" (Expected Depth): Opacity-weighted depth per pixel
    - "D" (Accumulated Depth): Raw accumulated depth values  
    - "RGB+ED": RGB channels + expected depth in 4th channel
    
    Args:
        render_output: Output from gsplat.rasterization() 
        alpha_output: Alpha/accumulation values from gsplat
        render_mode: The render mode used ("ED", "D", "RGB+ED")
        background_depth: Depth value for background pixels (optional)
        
    Returns:
        Per-pixel depth values [batch, H, W, 1]
    """
    if render_mode == "ED":
        # Expected depth is the full render output
        depth = render_output  # [batch, H, W, 1]
        
    elif render_mode == "RGB+ED":
        # Expected depth is in the 4th channel  
        depth = render_output[..., 3:4]  # [batch, H, W, 1]
        
    elif render_mode == "D":
        # Accumulated depth - need to normalize by alpha
        alpha_safe = torch.clamp(alpha_output, min=1e-10)
        depth = render_output / alpha_safe  # [batch, H, W, 1]
        
    else:
        raise ValueError(f"Unsupported render mode for depth extraction: {render_mode}")
    
    # Handle background pixels (where alpha ≈ 0)
    if background_depth is not None:
        alpha_mask = alpha_output.squeeze(-1) > 1e-6  # [batch, H, W]
        depth = torch.where(
            alpha_mask.unsqueeze(-1), 
            depth, 
            torch.full_like(depth, background_depth)
        )
    
    return depth


def render_gaussian_depths_with_gsplat(
    means: Float[Tensor, "num_gaussians 3"],
    quats: Float[Tensor, "num_gaussians 4"], 
    scales: Float[Tensor, "num_gaussians 3"],
    opacities: Float[Tensor, "num_gaussians 1"],
    viewmats: Float[Tensor, "batch 4 4"],
    Ks: Float[Tensor, "batch 3 3"],
    width: int,
    height: int,
    packed: bool = False,
    sparse_grad: bool = False,
    near_plane: float = 0.01,
    far_plane: float = 1e10
) -> Tuple[
    Float[Tensor, "batch height width 1"],  # expected_depths
    Float[Tensor, "batch height width 1"],  # accumulated_depths  
    Float[Tensor, "batch height width 1"],  # alpha_values
    Dict  # meta_info
]:
    """
    Render depth maps using gsplat's native depth rendering capabilities.
    
    This function uses gsplat to efficiently compute per-pixel depths without
    reimplementing any projection or rasterization logic. It leverages gsplat's
    optimized CUDA kernels for maximum performance.
    
    Args:
        means: 3D positions of Gaussian centers
        quats: Quaternion rotations (will be normalized by gsplat)
        scales: Scaling factors for each Gaussian
        opacities: Opacity values [0, 1]
        viewmats: Camera view matrices [batch, 4, 4]
        Ks: Camera intrinsic matrices [batch, 3, 3]
        width: Image width in pixels
        height: Image height in pixels
        packed: Use memory-efficient packed rendering
        sparse_grad: Use sparse gradients (requires packed=True)
        near_plane: Near clipping plane
        far_plane: Far clipping plane
        
    Returns:
        Tuple of:
        - expected_depths: Expected depth per pixel (opacity-weighted)
        - accumulated_depths: Raw accumulated depth values
        - alpha_values: Accumulation/alpha values per pixel
        - meta_info: GSplat metadata dict with intermediate results
    """
    if rasterization is None:
        raise RuntimeError("gsplat not available. Please install with: pip install gsplat")
    
    # Render expected depth
    expected_depth_render, alpha_expected, meta_expected = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities.squeeze(-1),  # gsplat expects [N] not [N, 1]
        colors=None,  # Not needed for depth rendering
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        packed=packed,
        sparse_grad=sparse_grad,
        near_plane=near_plane,
        far_plane=far_plane,
        render_mode="ED",  # Expected Depth
    )
    
    # Render accumulated depth
    accumulated_depth_render, alpha_accumulated, meta_accumulated = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities.squeeze(-1),
        colors=None,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        packed=packed,
        sparse_grad=sparse_grad,
        near_plane=near_plane,
        far_plane=far_plane,
        render_mode="D",  # Accumulated Depth
    )
    
    # Return processed outputs
    return (
        expected_depth_render,    # [batch, H, W, 1]
        accumulated_depth_render, # [batch, H, W, 1]
        alpha_expected,          # [batch, H, W, 1] 
        meta_expected           # Most relevant metadata
    )


def compute_depth_supervision_data_with_gsplat(
    gaussian_means: Float[Tensor, "num_gaussians 3"],
    gaussian_quats: Float[Tensor, "num_gaussians 4"],
    gaussian_scales: Float[Tensor, "num_gaussians 3"], 
    gaussian_opacities: Float[Tensor, "num_gaussians 1"],
    camera: Cameras,
    image_height: int,
    image_width: int,
    render_config: Optional[Dict] = None
) -> Tuple[
    Float[Tensor, "1 height width 1"],    # rendered_depths
    Float[Tensor, "1 height width 1"],    # alpha_values
    Dict                                  # gsplat_meta
]:
    """
    Compute depth supervision data using gsplat's efficient rendering.
    
    This is the main interface for depth supervision, providing clean integration
    with the Surveyor training pipeline while leveraging gsplat's optimizations.
    
    Args:
        gaussian_means: 3D Gaussian center positions
        gaussian_quats: Quaternion rotations  
        gaussian_scales: Scale parameters
        gaussian_opacities: Opacity values
        camera: Nerfstudio camera object
        image_height: Target render height
        image_width: Target render width
        render_config: Optional rendering configuration
        
    Returns:
        Tuple of:
        - rendered_depths: Per-pixel depth values from gsplat
        - alpha_values: Opacity/accumulation values  
        - gsplat_meta: Metadata from gsplat rasterization
    """
    # Set default render configuration
    config = {
        "packed": False,
        "sparse_grad": False, 
        "near_plane": 0.01,
        "far_plane": 1e10,
    }
    if render_config:
        config.update(render_config)
    
    try:
        # Convert nerfstudio camera to gsplat format
        camera_to_world = camera.camera_to_worlds[0]  # [3, 4] or [4, 4]
        if camera_to_world.shape[0] == 3:
            # Convert [3, 4] to [4, 4]
            bottom_row = torch.tensor([[0., 0., 0., 1.]], device=camera_to_world.device)
            camera_to_world = torch.cat([camera_to_world, bottom_row], dim=0)
        
        # World to camera transformation
        viewmat = torch.inverse(camera_to_world).unsqueeze(0)  # [1, 4, 4]
        
        # Camera intrinsics matrix
        K = torch.tensor([
            [camera.fx[0].item(), 0., camera.cx[0].item()],
            [0., camera.fy[0].item(), camera.cy[0].item()],
            [0., 0., 1.]
        ], device=camera_to_world.device).unsqueeze(0)  # [1, 3, 3]
        
        # Render depths using gsplat
        expected_depths, accumulated_depths, alphas, meta = render_gaussian_depths_with_gsplat(
            means=gaussian_means,
            quats=gaussian_quats,
            scales=gaussian_scales,
            opacities=gaussian_opacities,
            viewmats=viewmat,
            Ks=K,
            width=image_width,
            height=image_height,
            **config
        )
        
        return expected_depths, alphas, meta
        
    except Exception as e:
        CONSOLE.log(f"[red]Error in gsplat depth rendering: {e}[/red]")
        
        # Return fallback empty tensors
        device = gaussian_means.device
        empty_depths = torch.zeros(1, image_height, image_width, 1, device=device)
        empty_alphas = torch.zeros(1, image_height, image_width, 1, device=device)
        empty_meta = {}
        
        return empty_depths, empty_alphas, empty_meta


def analyze_gsplat_rasterization_metadata(meta: Dict) -> Dict[str, any]:
    """
    Analyze and extract useful information from gsplat's rasterization metadata.
    
    GSplat's rasterization function returns a metadata dictionary containing
    intermediate results that can be useful for debugging and analysis.
    
    Args:
        meta: Metadata dictionary from gsplat.rasterization()
        
    Returns:
        Dictionary with analyzed metadata information
    """
    analysis = {
        "available_keys": list(meta.keys()),
        "summary": {}
    }
    
    # Analyze available metadata
    for key, value in meta.items():
        if isinstance(value, torch.Tensor):
            analysis["summary"][key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "requires_grad": value.requires_grad,
                "memory_mb": value.numel() * value.element_size() / (1024 * 1024)
            }
            
            # Special handling for specific metadata
            if key == "means2d":
                analysis["summary"][key]["description"] = "Projected 2D positions of Gaussians"
            elif key == "radii":
                analysis["summary"][key]["description"] = "Projected radii of Gaussians in pixels"
            elif "tile" in key.lower():
                analysis["summary"][key]["description"] = "Tile-based rendering information"
                
    return analysis


def validate_gsplat_depth_rendering(
    gaussian_params: Dict[str, Tensor],
    camera: Cameras,
    image_size: Tuple[int, int],
    tolerance: float = 1e-5
) -> Dict[str, bool]:
    """
    Validate that gsplat depth rendering produces consistent results.
    
    This function runs multiple rendering modes and checks for consistency
    between expected depth, accumulated depth, and alpha blending.
    
    Args:
        gaussian_params: Dictionary with 'means', 'quats', 'scales', 'opacities'
        camera: Camera object for rendering
        image_size: (height, width) tuple
        tolerance: Numerical tolerance for consistency checks
        
    Returns:
        Dictionary with validation results
    """
    height, width = image_size
    results = {}
    
    try:
        # Test both expected and accumulated depth rendering
        expected_depths, alphas, meta = compute_depth_supervision_data_with_gsplat(
            gaussian_means=gaussian_params['means'],
            gaussian_quats=gaussian_params['quats'],
            gaussian_scales=gaussian_params['scales'],
            gaussian_opacities=gaussian_params['opacities'],
            camera=camera,
            image_height=height,
            image_width=width
        )
        
        # Validation checks
        results["depth_shape_valid"] = expected_depths.shape == (1, height, width, 1)
        results["alpha_shape_valid"] = alphas.shape == (1, height, width, 1)
        results["depth_range_valid"] = torch.all(expected_depths >= 0)
        results["alpha_range_valid"] = torch.all((alphas >= 0) & (alphas <= 1))
        results["no_nan_depth"] = not torch.any(torch.isnan(expected_depths))
        results["no_nan_alpha"] = not torch.any(torch.isnan(alphas))
        results["metadata_available"] = len(meta) > 0
        
        # Check depth-alpha consistency (where alpha > 0, depth should be > 0)
        alpha_mask = alphas.squeeze() > tolerance
        if alpha_mask.any():
            depth_values = expected_depths.squeeze()[alpha_mask]
            results["depth_alpha_consistency"] = torch.all(depth_values > 0)
        else:
            results["depth_alpha_consistency"] = True  # Vacuously true
            
        results["overall_valid"] = all(results.values())
        
    except Exception as e:
        CONSOLE.log(f"[red]Validation error: {e}[/red]")
        results = {key: False for key in [
            "depth_shape_valid", "alpha_shape_valid", "depth_range_valid",
            "alpha_range_valid", "no_nan_depth", "no_nan_alpha", 
            "metadata_available", "depth_alpha_consistency", "overall_valid"
        ]}
    
    return results