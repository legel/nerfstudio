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
Depth supervision losses for Gaussian Splatting in Surveyor project.

This module implements depth supervision for 3D Gaussian Splats, adapting the 
DS-NeRF approach to the discrete nature of Gaussian Splatting rendering.

Key Differences from NeRF Depth Supervision:
────────────────────────────────────────────

NeRF (Continuous):
    Ray → Samples → Densities → Volume Rendering → Depth Loss
           ↓          ↓            ↓                ↓
       Structured   Continuous   Alpha composite   KL divergence
       sampling     field        along ray         on samples

Gaussian Splatting (Discrete):
    Splats → Rasterization → Image → Depth Loss  
      ↓         ↓            ↓        ↓
    Irregular  Project to   Pixel    Splat-based
    positions  pixels       colors   supervision

Mathematical Foundation:
───────────────────────

DS-NeRF Loss (per ray):
    L_depth = -log(Σ w_i) * exp(-((t_i - d_sensor)² / (2σ²))) * δt_i

Splat Depth Loss (per pixel):
    L_splat = -log(Σ α_j) * exp(-((z_j - d_sensor)² / (2σ²))) * A_j

Where:
    α_j: Opacity of splat j contributing to pixel
    z_j: Distance from camera center to splat j  
    d_sensor: Ground truth depth from depth sensor
    σ: Depth uncertainty parameter
    A_j: Area contribution of splat j to pixel

ASCII Diagram - Depth Supervision Concept:
──────────────────────────────────────────

    Camera ●────────► Pixel
             \
              \  d_sensor (ground truth)
               \
                ●  True surface
              / | \
             /  |  \  Multiple splats at different depths
           ●    ●    ●
          z_1  z_2  z_3  (splat depths)
          α_1  α_2  α_3  (splat opacities)

Goal: Penalize splats that deviate from ground truth depth while
      preserving splats that align with sensor measurements.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from jaxtyping import Float
from torch import Tensor

# Epsilon for numerical stability  
EPS = 1.0e-7

def get_depth_loss_weight(step: int) -> float:
    """
    Compute depth loss weight based on training step for splat sculpting.
    
    Strategy:
    - Early training (0-5000): High weight for geometric sculpting
    - Later training (5000+): Gradual decay to preserve color refinement
    
    ASCII Schedule Visualization:
    
    Weight
    1.0 ├─╲
        │  ╲
    0.5 │   ╲───╲
        │        ╲───╲  
    0.1 │             ╲───╲
        │                  ╲─────────
    0.0 └────┴────┴────┴────┴─────────► Step
        0   1k   2k   5k   10k   20k
    
    Args:
        step: Current training step
        
    Returns:
        Depth loss weight multiplier
    """
    if step < 5000:
        # High initial weight, linear decay to 0.5
        return 1.0 - 0.5 * (step / 5000.0)
    elif step < 15000:
        # Gradual decay from 0.5 to 0.01
        progress = (step - 5000) / 10000.0
        return 0.5 * (1.0 - progress) + 0.01 * progress
    else:
        # Maintain minimal supervision
        return 0.01


def extract_splat_depths_from_rasterization(
    raster_output: Dict[str, Tensor],
    camera_centers: Float[Tensor, "3"]
) -> Tuple[
    Float[Tensor, "height width max_splats"], 
    Float[Tensor, "height width max_splats"],
    Float[Tensor, "height width"]
]:
    """
    Extract per-pixel splat depth information from rasterization output.
    
    This function processes the rasterization results to identify which
    3D Gaussian splats contribute to each pixel and their distances from
    the camera center.
    
    Args:
        raster_output: Rasterization output dictionary containing:
            - 'splat_ids': [H, W, max_splats] indices of contributing splats
            - 'splat_opacities': [H, W, max_splats] opacity contributions
            - 'splat_positions': [N_splats, 3] world positions of all splats
        camera_centers: [3] camera center position in world coordinates
        
    Returns:
        Tuple of:
        - splat_depths: [H, W, max_splats] distances from camera to each splat
        - splat_opacities: [H, W, max_splats] opacity of each contributing splat
        - splat_counts: [H, W] number of contributing splats per pixel
        
    Note:
        This function assumes rasterization provides splat contribution info.
        If gsplat doesn't provide this, we need to modify the rasterization
        or implement a custom gathering mechanism.
    """
    # Extract contributing splat information
    splat_ids = raster_output.get('splat_ids')
    splat_opacities = raster_output.get('splat_opacities') 
    splat_positions = raster_output.get('splat_positions')
    
    if splat_ids is None or splat_opacities is None or splat_positions is None:
        raise ValueError(
            "Rasterization output must contain 'splat_ids', 'splat_opacities', "
            "and 'splat_positions' for depth supervision. "
            "Consider modifying gsplat rasterization to provide this information."
        )
    
    # Gather splat positions for contributing splats
    H, W, max_splats = splat_ids.shape
    device = splat_ids.device
    
    # Handle invalid splat IDs (padding)
    valid_mask = splat_ids >= 0
    
    # Clamp splat IDs to valid range for gathering
    clamped_ids = torch.clamp(splat_ids, min=0)
    
    # Gather world positions: [H, W, max_splats, 3]
    gathered_positions = splat_positions[clamped_ids]
    
    # Compute distances from camera center: [H, W, max_splats]
    camera_centers_expanded = camera_centers[None, None, None, :]  # [1, 1, 1, 3]
    splat_distances = torch.norm(
        gathered_positions - camera_centers_expanded, 
        dim=-1
    )
    
    # Mask out invalid splats
    splat_depths = torch.where(valid_mask, splat_distances, torch.tensor(0.0, device=device))
    splat_opacities_masked = torch.where(valid_mask, splat_opacities, torch.tensor(0.0, device=device))
    
    # Count contributing splats per pixel
    splat_counts = valid_mask.sum(dim=-1).float()
    
    return splat_depths, splat_opacities_masked, splat_counts


def splat_depth_loss(
    splat_depths: Float[Tensor, "height width max_splats"],
    splat_opacities: Float[Tensor, "height width max_splats"], 
    sensor_depths: Float[Tensor, "height width"],
    depth_valid_mask: Float[Tensor, "height width"],
    sigma: float = 0.01
) -> Float[Tensor, ""]:
    """
    Compute depth supervision loss for Gaussian Splats based on DS-NeRF approach.
    
    This implements the core depth supervision loss adapted for the discrete
    nature of Gaussian Splatting. Instead of continuous ray samples, we work
    with discrete splats that contribute to each pixel.
    
    Mathematical Formulation:
    ────────────────────────
    
    For each pixel with valid depth supervision:
    
    L_pixel = -log(Σ α_j + ε) * Σ [α_j * exp(-((z_j - d_sensor)² / (2σ²)))]
    
    Where the sum is over all contributing splats j at that pixel.
    
    ASCII Illustration:
    ──────────────────
    
    Camera ●──────────► Pixel (u,v)
             \
              \ d_sensor = 2.5m
               \
                ●  True surface
              / | \
            1.8m 2.5m 3.2m  ← Splat depths z_j
            α₁   α₂   α₃    ← Splat opacities
            
    Loss encourages:
    - Splat at 2.5m: High opacity (matches sensor)
    - Splats at 1.8m, 3.2m: Lower opacity (deviate from sensor)
    
    Args:
        splat_depths: [H, W, max_splats] distances from camera to contributing splats
        splat_opacities: [H, W, max_splats] opacity values of contributing splats
        sensor_depths: [H, W] ground truth depths from depth sensor (meters)
        depth_valid_mask: [H, W] boolean mask of pixels with valid depth
        sigma: Depth uncertainty parameter (default: 1cm)
        
    Returns:
        Scalar depth supervision loss
    """
    # Only supervise pixels with valid depth measurements
    if not depth_valid_mask.any():
        return torch.tensor(0.0, device=splat_depths.device, requires_grad=True)
    
    # Extract valid pixels for supervision
    valid_splat_depths = splat_depths[depth_valid_mask]      # [N_valid, max_splats]
    valid_splat_opacities = splat_opacities[depth_valid_mask] # [N_valid, max_splats]
    valid_sensor_depths = sensor_depths[depth_valid_mask]    # [N_valid]
    
    # Compute depth differences for each splat
    # Shape: [N_valid, max_splats]
    depth_diff = valid_splat_depths - valid_sensor_depths[:, None]
    
    # Gaussian weighting based on depth deviation
    # Splats close to sensor depth get higher weight
    depth_weights = torch.exp(-(depth_diff ** 2) / (2 * sigma ** 2))
    
    # Combine opacity and depth weighting
    # Shape: [N_valid, max_splats]
    weighted_opacities = valid_splat_opacities * depth_weights
    
    # Sum contributions per pixel
    # Shape: [N_valid]
    total_weighted_opacity = weighted_opacities.sum(dim=-1)
    total_opacity = valid_splat_opacities.sum(dim=-1)
    
    # DS-NeRF inspired loss: -log(total_opacity) * weighted_contribution
    # Add epsilon for numerical stability
    loss_per_pixel = -torch.log(total_opacity + EPS) * total_weighted_opacity
    
    # Average over all supervised pixels
    return loss_per_pixel.mean()


def sparse_splat_depth_loss(
    splat_depths: Float[Tensor, "height width max_splats"],
    splat_opacities: Float[Tensor, "height width max_splats"],
    sensor_depths: Float[Tensor, "height width"], 
    depth_valid_mask: Float[Tensor, "height width"],
    confidence_threshold: float = 0.01,
    sigma: float = 0.01
) -> Float[Tensor, ""]:
    """
    Memory-efficient sparse depth supervision that only processes pixels with:
    1. Valid depth sensor readings
    2. Contributing splats (total opacity > threshold)
    3. High confidence depth measurements
    
    This addresses the sparse nature of Gaussian Splatting where many pixels
    may have no contributing splats, especially early in training.
    
    Args:
        splat_depths: [H, W, max_splats] depths of contributing splats
        splat_opacities: [H, W, max_splats] opacities of contributing splats
        sensor_depths: [H, W] ground truth depth measurements  
        depth_valid_mask: [H, W] validity mask for depth measurements
        confidence_threshold: Minimum total opacity to apply supervision
        sigma: Depth uncertainty parameter
        
    Returns:
        Scalar sparse depth supervision loss
    """
    # Find pixels with both valid depth and contributing splats
    total_opacity = splat_opacities.sum(dim=-1)
    has_splats = total_opacity > confidence_threshold
    
    # Combined mask: valid depth AND contributing splats
    supervision_mask = depth_valid_mask & has_splats
    
    if not supervision_mask.any():
        return torch.tensor(0.0, device=splat_depths.device, requires_grad=True)
    
    # Apply standard depth loss to supervised pixels
    return splat_depth_loss(
        splat_depths, 
        splat_opacities, 
        sensor_depths, 
        supervision_mask, 
        sigma
    )


def create_depth_supervision_test_visualization(
    splat_depths: Float[Tensor, "height width max_splats"],
    sensor_depths: Float[Tensor, "height width"],
    depth_valid_mask: Float[Tensor, "height width"],
    output_path: str
) -> None:
    """
    Create visualization of depth supervision for debugging and validation.
    
    Generates a side-by-side comparison of:
    - Sensor depth map (ground truth)
    - Splat depth map (rendered from Gaussians)  
    - Depth difference map (error visualization)
    
    Args:
        splat_depths: [H, W, max_splats] depths from splats
        sensor_depths: [H, W] ground truth depths
        depth_valid_mask: [H, W] validity mask
        output_path: Path to save visualization image
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Compute expected depth from splats (closest non-zero)
    valid_splat_mask = splat_depths > 0
    closest_splat_depths = torch.where(
        valid_splat_mask.any(dim=-1),
        splat_depths.masked_fill(~valid_splat_mask, float('inf')).min(dim=-1)[0],
        torch.tensor(0.0)
    )
    
    # Convert to numpy for visualization
    sensor_np = sensor_depths.detach().cpu().numpy()
    splat_np = closest_splat_depths.detach().cpu().numpy()
    valid_np = depth_valid_mask.detach().cpu().numpy()
    
    # Compute difference where both are valid
    diff_np = np.where(valid_np & (splat_np > 0), 
                      np.abs(sensor_np - splat_np), 0)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(sensor_np, cmap='viridis')
    axes[0].set_title('Sensor Depth (Ground Truth)')
    axes[0].axis('off')
    
    axes[1].imshow(splat_np, cmap='viridis') 
    axes[1].set_title('Splat Depth (Rendered)')
    axes[1].axis('off')
    
    im = axes[2].imshow(diff_np, cmap='hot')
    axes[2].set_title('Depth Error (|Sensor - Splat|)')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], label='Error (meters)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()