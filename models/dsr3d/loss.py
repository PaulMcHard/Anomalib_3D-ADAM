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
        batch: Dict[str, torch.Tensor],
        training_phase: int = 1,
    ) -> torch.Tensor:
        """Compute loss based on training phase.
        
        Args:
            outputs: Model outputs containing various predictions and losses
            batch: Input batch with ground truth data
            training_phase: Current training phase (1, 2, or 3)
            
        Returns:
            Computed loss
        """
        if training_phase == 1:
            return self._phase_1_loss(outputs, batch)
        elif training_phase == 2:
            return self._phase_2_loss(outputs, batch)
        else:  # training_phase == 3
            return self._phase_3_loss(outputs, batch)
    
    def _phase_1_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Phase 1: VQ-VAE reconstruction and commitment loss.
        
        This phase trains:
        - The discrete latent model (VQ-VAE)
        - General appearance decoder
        - Object-specific decoder (subspace restriction)
        """
        # Get input data (handle both depth-only and RGB+depth modes)
        if "depth" in batch:
            # RGB+Depth mode: concatenate depth first, then RGB
            input_data = torch.cat([batch["depth"], batch["image"]], dim=1)
        else:
            # Depth-only mode or fallback
            input_data = batch["image"]
        
        total_loss = 0.0
        
        # VQ losses (from the discrete latent model)
        if "vq_loss_top" in outputs:
            vq_loss_top = outputs["vq_loss_top"]
            total_loss += vq_loss_top
            
        if "vq_loss_bottom" in outputs:
            vq_loss_bottom = outputs["vq_loss_bottom"] 
            total_loss += vq_loss_bottom
        
        # General reconstruction loss
        if "general_reconstruction" in outputs:
            general_recon_loss = self.mse_loss(
                outputs["general_reconstruction"], 
                input_data
            )
            total_loss += general_recon_loss
        
        # Object-specific reconstruction loss (subspace restriction)
        if "object_specific_reconstruction" in outputs:
            object_recon_loss = self.mse_loss(
                outputs["object_specific_reconstruction"],
                input_data
            )
            total_loss += object_recon_loss
        
        return total_loss
    
    def _phase_2_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Phase 2: Anomaly detection loss with synthetic anomalies.
        
        This phase trains the anomaly detection module using:
        - Synthetic anomalies generated in latent space
        - Real normal images as negative examples
        """
        if "anomaly_logits" not in outputs:
            # Fallback to reconstruction loss if anomaly detection not available
            return self._reconstruction_error_loss(outputs, batch)
        
        anomaly_logits = outputs["anomaly_logits"]  # Shape: [B, 2, H, W]
        
        if "mask" in batch and batch["mask"] is not None:
            # Supervised training with ground truth masks
            target_mask = batch["mask"].float()
            
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
            # The model should generate synthetic anomalies during training
            # and learn to distinguish them from normal regions
            
            # For unsupervised case, we assume the model generates synthetic anomalies
            # and the loss is computed internally. Here we can add a consistency loss
            # or other unsupervised objectives.
            
            # Simple consistency loss between normal and anomaly channels
            normal_channel = anomaly_logits[:, 0:1]
            anomaly_channel = anomaly_logits[:, 1:2]
            
            # Encourage normal regions to have high normal probability
            # and low anomaly probability
            normal_loss = -torch.mean(normal_channel)  # Maximize normal channel
            anomaly_reg = torch.mean(torch.abs(anomaly_channel))  # Minimize anomaly channel for normal data
            
            return normal_loss + 0.1 * anomaly_reg
    
    def _phase_3_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Phase 3: Upsampling module loss for final anomaly maps.
        
        This phase trains the upsampling module to produce high-quality
        anomaly maps from the coarse anomaly detection outputs.
        """
        if "anomaly_map" not in outputs:
            # Fallback if no anomaly map available
            return torch.tensor(0.0, device=next(iter(outputs.values())).device)
        
        anomaly_map = outputs["anomaly_map"]
        
        if "mask" in batch and batch["mask"] is not None:
            # Supervised training with ground truth masks
            target_mask = batch["mask"].float()
            
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
            # Unsupervised case: use smoothness and consistency losses
            # Smoothness loss to encourage coherent regions
            smoothness_loss = self._smoothness_loss(anomaly_map)
            
            # Consistency with previous phase outputs
            if "anomaly_logits" in outputs:
                # Ensure consistency between upsampled map and raw logits
                raw_anomaly = torch.sigmoid(outputs["anomaly_logits"][:, 1:2])
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
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Fallback reconstruction error loss."""
        if "object_specific_reconstruction" in outputs:
            if "depth" in batch:
                input_data = torch.cat([batch["depth"], batch["image"]], dim=1)
            else:
                input_data = batch["image"]
                
            return self.mse_loss(outputs["object_specific_reconstruction"], input_data)
        
        return torch.tensor(0.0, device=next(iter(outputs.values())).device)