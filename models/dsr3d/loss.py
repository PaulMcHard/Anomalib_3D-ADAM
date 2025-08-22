"""3DSR Loss Functions."""

# Copyright (C) 2025 Intel Corporation  
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

__all__ = ["Dsr3dLoss", "FocalLoss"]


class FocalLoss(nn.Module):
    """Focal Loss for anomaly segmentation."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss."""
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class Dsr3dLoss(nn.Module):
    """Loss function for 3DSR model.
    
    Implements the multi-phase training loss as described in the 3DSR paper.
    
    Phase 1: VQ-VAE reconstruction loss + VQ commitment loss
    Phase 2: Anomaly detection loss with synthetic anomalies  
    Phase 3: Upsampling module loss for final anomaly maps
    """
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.focal_loss = FocalLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch,  # Can be dict or anomalib Batch object
        training_phase: int = 1,
    ) -> torch.Tensor:
        """Compute loss based on training phase."""
        if training_phase == 1:
            return self._phase_1_loss(outputs, batch)
        elif training_phase == 2:
            return self._phase_2_loss(outputs, batch)
        else:  # training_phase == 3
            return self._phase_3_loss(outputs, batch)
    
    def _phase_1_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch,  # Can be dict or anomalib Batch object
    ) -> torch.Tensor:
        """Phase 1: VQ-VAE reconstruction and commitment loss."""
        # Handle both dict and anomalib Batch objects
        if hasattr(batch, 'image'):
            # Anomalib Batch object
            rgb_data = batch.image
            depth_data = getattr(batch, 'depth', None)
        else:
            # Dictionary format
            rgb_data = batch["image"]
            depth_data = batch.get("depth", None)
        
        # Get input data (handle both depth-only and RGB+depth modes)
        if depth_data is not None:
            # RGB+Depth mode: concatenate depth first, then RGB
            input_data = torch.cat([depth_data, rgb_data], dim=1)
        else:
            # Handle different RGB formats
            if rgb_data.shape[1] == 4:
                # RGBD format - split into RGB and D
                depth_data = rgb_data[:, :1]  # First channel as depth
                rgb_data = rgb_data[:, 1:]    # Remaining channels as RGB
                input_data = torch.cat([depth_data, rgb_data], dim=1)
            elif rgb_data.shape[1] == 3:
                # RGB only - create dummy depth channel
                depth_data = torch.zeros_like(rgb_data[:, :1])
                input_data = torch.cat([depth_data, rgb_data], dim=1)
            else:
                # Single channel or other format
                input_data = rgb_data
        
        total_loss = 0.0
        
        # VQ losses (from the discrete latent model)
        if "vq_loss_top" in outputs:
            vq_loss_top = outputs["vq_loss_top"]
            total_loss += vq_loss_top
            
        if "vq_loss_bottom" in outputs:
            vq_loss_bottom = outputs["vq_loss_bottom"] 
            total_loss += vq_loss_bottom
        
        # General reconstruction loss - handle channel and size mismatch
        if "general_reconstruction" in outputs:
            general_reconstruction = outputs["general_reconstruction"]
            
            # Handle channel mismatch
            if general_reconstruction.shape[1] != input_data.shape[1]:
                if general_reconstruction.shape[1] == 3 and input_data.shape[1] == 4:
                    # Compare only RGB channels (skip depth channel at index 0)
                    target_for_general = input_data[:, 1:]  # Skip depth channel
                elif general_reconstruction.shape[1] == 1 and input_data.shape[1] == 4:
                    # Compare only depth channel
                    target_for_general = input_data[:, :1]  # Only depth channel
                else:
                    # Use minimum channels
                    min_channels = min(general_reconstruction.shape[1], input_data.shape[1])
                    general_reconstruction = general_reconstruction[:, :min_channels]
                    target_for_general = input_data[:, :min_channels]
            else:
                target_for_general = input_data
            
            # Handle size mismatch
            if general_reconstruction.shape[-2:] != target_for_general.shape[-2:]:
                general_reconstruction = F.interpolate(
                    general_reconstruction,
                    size=target_for_general.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            general_recon_loss = self.mse_loss(general_reconstruction, target_for_general)
            total_loss += general_recon_loss
        
        # Object-specific reconstruction loss - handle channel and size mismatch
        if "object_specific_reconstruction" in outputs:
            object_reconstruction = outputs["object_specific_reconstruction"]
            
            # Handle channel mismatch
            if object_reconstruction.shape[1] != input_data.shape[1]:
                if object_reconstruction.shape[1] == 3 and input_data.shape[1] == 4:
                    # Compare only RGB channels (skip depth)
                    target_for_object = input_data[:, 1:]  # Skip depth channel
                elif object_reconstruction.shape[1] == 1 and input_data.shape[1] == 4:
                    # Compare only depth channel
                    target_for_object = input_data[:, :1]  # Only depth channel
                else:
                    # Use minimum channels
                    min_channels = min(object_reconstruction.shape[1], input_data.shape[1])
                    object_reconstruction = object_reconstruction[:, :min_channels]
                    target_for_object = input_data[:, :min_channels]
            else:
                target_for_object = input_data
            
            # Handle size mismatch
            if object_reconstruction.shape[-2:] != target_for_object.shape[-2:]:
                object_reconstruction = F.interpolate(
                    object_reconstruction,
                    size=target_for_object.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            object_recon_loss = self.mse_loss(object_reconstruction, target_for_object)
            total_loss += object_recon_loss
        
        return total_loss
    
    def _phase_2_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch,  # Can be dict or anomalib Batch object
    ) -> torch.Tensor:
        """Phase 2: Anomaly detection loss with synthetic anomalies."""
        if "anomaly_logits" not in outputs:
            # Fallback to reconstruction loss if anomaly detection not available
            return self._reconstruction_error_loss(outputs, batch)
        
        anomaly_logits = outputs["anomaly_logits"]  # Shape: [B, 2, H, W]
        
        # Handle both dict and anomalib Batch objects for mask
        if hasattr(batch, 'mask'):
            mask = batch.mask
        elif isinstance(batch, dict) and "mask" in batch:
            mask = batch["mask"]
        else:
            mask = None
            
        if mask is not None:
            # Supervised training with ground truth masks
            target_mask = mask.float()
            
            # Reshape target to match logits if needed
            if target_mask.dim() == 3:
                target_mask = target_mask.unsqueeze(1)
            if target_mask.shape[1] != 1:
                target_mask = target_mask[:, :1]  # Take first channel
                
            # Resize target mask to match anomaly logits size
            if target_mask.shape[-2:] != anomaly_logits.shape[-2:]:
                target_mask = F.interpolate(
                    target_mask,
                    size=anomaly_logits.shape[-2:],
                    mode='nearest'
                )
            
            # Use focal loss for anomaly detection (class 1 = anomaly)
            anomaly_channel = anomaly_logits[:, 1:2]  # Take anomaly channel
            focal_loss = self.focal_loss(anomaly_channel, target_mask)
            
            return focal_loss
            
        else:
            # Unsupervised training with synthetic anomalies
            normal_channel = anomaly_logits[:, 0:1]
            anomaly_channel = anomaly_logits[:, 1:2]
            
            # Encourage normal regions to have high normal probability
            normal_loss = -torch.mean(normal_channel)
            anomaly_reg = torch.mean(torch.abs(anomaly_channel))
            
            return normal_loss + 0.1 * anomaly_reg
    
    def _phase_3_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch,  # Can be dict or anomalib Batch object
    ) -> torch.Tensor:
        """Phase 3: Upsampling module loss for final anomaly maps."""
        if "anomaly_map" not in outputs:
            return torch.tensor(0.0, device=next(iter(outputs.values())).device)
        
        anomaly_map = outputs["anomaly_map"]
        
        # Handle both dict and anomalib Batch objects for mask
        if hasattr(batch, 'mask'):
            mask = batch.mask
        elif isinstance(batch, dict) and "mask" in batch:
            mask = batch["mask"]
        else:
            mask = None
            
        if mask is not None:
            # Supervised training with ground truth masks
            target_mask = mask.float()
            
            # Ensure target mask has correct dimensions
            if target_mask.dim() == 3:
                target_mask = target_mask.unsqueeze(1)
            if target_mask.shape[1] != 1:
                target_mask = target_mask[:, :1]
                
            # Resize target to match anomaly map size
            if target_mask.shape[-2:] != anomaly_map.shape[-2:]:
                target_mask = F.interpolate(
                    target_mask,
                    size=anomaly_map.shape[-2:],
                    mode='nearest'
                )
            
            # Use MSE loss for upsampling (since anomaly_map is already sigmoid activated)
            upsampling_loss = self.mse_loss(anomaly_map, target_mask)
            
            return upsampling_loss
            
        else:
            # Unsupervised case: use smoothness loss
            smoothness_loss = self._smoothness_loss(anomaly_map)
            
            # Consistency with previous phase outputs
            if "anomaly_logits" in outputs:
                raw_anomaly = torch.sigmoid(outputs["anomaly_logits"][:, 1:2])
                
                # Ensure both tensors have the same spatial dimensions
                if raw_anomaly.shape[-2:] != anomaly_map.shape[-2:]:
                    raw_anomaly = F.interpolate(
                        raw_anomaly,
                        size=anomaly_map.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                consistency_loss = self.mse_loss(anomaly_map, raw_anomaly)
                return smoothness_loss + 0.5 * consistency_loss
            
            return smoothness_loss
    
    def _smoothness_loss(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """Compute smoothness loss to encourage coherent anomaly regions."""
        # Gradient in x and y directions
        grad_x = torch.abs(anomaly_map[:, :, :, 1:] - anomaly_map[:, :, :, :-1])
        grad_y = torch.abs(anomaly_map[:, :, 1:, :] - anomaly_map[:, :, :-1, :])
        
        return torch.mean(grad_x) + torch.mean(grad_y)
    
    def _reconstruction_error_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch,  # Can be dict or anomalib Batch object
    ) -> torch.Tensor:
        """Fallback reconstruction error loss."""
        if "object_specific_reconstruction" in outputs:
            # Handle both dict and anomalib Batch objects
            if hasattr(batch, 'image'):
                rgb_data = batch.image
                depth_data = getattr(batch, 'depth', None)
            else:
                rgb_data = batch["image"]
                depth_data = batch.get("depth", None)
                
            if depth_data is not None:
                input_data = torch.cat([depth_data, rgb_data], dim=1)
            else:
                input_data = rgb_data
            
            object_reconstruction = outputs["object_specific_reconstruction"]
            
            # Handle channel mismatch
            if object_reconstruction.shape[1] != input_data.shape[1]:
                if object_reconstruction.shape[1] == 3 and input_data.shape[1] == 4:
                    target_data = input_data[:, 1:]  # RGB only
                elif object_reconstruction.shape[1] == 1 and input_data.shape[1] == 4:
                    target_data = input_data[:, :1]  # Depth only
                else:
                    target_data = input_data
            else:
                target_data = input_data
            
            # Handle size mismatch
            if object_reconstruction.shape[-2:] != target_data.shape[-2:]:
                object_reconstruction = F.interpolate(
                    object_reconstruction,
                    size=target_data.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                
            return self.mse_loss(object_reconstruction, target_data)
        
        return torch.tensor(0.0, device=next(iter(outputs.values())).device)