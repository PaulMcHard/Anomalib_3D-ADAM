"""3DSR Lightning Model Implementation."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Dict, Any, Optional, Union

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn

from anomalib.models.components import AnomalyModule
from .torch_model import Dsr3dModel
from .loss import Dsr3dLoss

logger = logging.getLogger(__name__)

__all__ = ["Dsr3d"]


class Dsr3d(AnomalyModule):
    """3DSR Lightning Module for 3D Surface Anomaly Detection.
    
    Args:
        rgb_channels: Number of RGB input channels
        depth_channels: Number of depth input channels  
        use_depth_only: Whether to use depth-only mode
        pretrained_vq_model_path: Path to pretrained VQ model
        latent_anomaly_strength: Strength of synthetic anomalies
        upsampling_train_ratio: Ratio for upsampling training
        num_hiddens: Number of hidden units in VQ-VAE
        num_residual_layers: Number of residual layers
        num_residual_hiddens: Number of residual hidden units
        num_embeddings: Size of quantization codebook
        embedding_dim: Dimension of quantization embeddings
        commitment_cost: Commitment cost for VQ loss
        decay: Decay rate for EMA quantization
        training_phase: Current training phase (1, 2, or 3)
        lr: Learning rate
        weight_decay: Weight decay for optimizer
    """
    
    def __init__(
        self,
        rgb_channels: int = 3,
        depth_channels: int = 1,
        use_depth_only: bool = False,
        pretrained_vq_model_path: Optional[str] = None,
        latent_anomaly_strength: float = 0.2,
        upsampling_train_ratio: float = 0.7,
        num_hiddens: int = 128,
        num_residual_layers: int = 2,
        num_residual_hiddens: int = 32,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        training_phase: int = 1,
        lr: float = 0.0002,
        weight_decay: float = 0.0001,
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.model = Dsr3dModel(
            rgb_channels=rgb_channels,
            depth_channels=depth_channels,
            use_depth_only=use_depth_only,
            pretrained_vq_model_path=pretrained_vq_model_path,
            latent_anomaly_strength=latent_anomaly_strength,
            upsampling_train_ratio=upsampling_train_ratio,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            decay=decay,
        )
        
        self.loss_fn = Dsr3dLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Set initial training phase
        self.set_training_phase(training_phase)
        
        # Track epochs for phase transitions (based on 3DSR paper recommendations)
        self.phase_1_epochs = 40  # VQ-VAE + Reconstruction (reduced due to pretrained VQ)
        self.phase_2_epochs = 50  # Anomaly Detection (most important phase)
        self.phase_3_epochs = 30  # Upsampling (fine-tuning)
        
    def set_training_phase(self, phase: int) -> None:
        """Set the training phase and update model accordingly."""
        self.model.set_training_phase(phase)
        self.current_phase = phase
        logger.info(f"Set training phase to {phase}")
        
    def configure_optimizers(self):
        """Configure optimizers for different training phases."""
        # Get trainable parameters based on current training phase
        if self.current_phase == 1:
            # Phase 1: Train VQ model and reconstruction components
            params = []
            params.extend(list(self.model.discrete_model.parameters()))
            params.extend(list(self.model.object_specific_decoder.parameters()))
            
        elif self.current_phase == 2:
            # Phase 2: Train anomaly detection module
            params = list(self.model.anomaly_detection_module.parameters())
            
        else:  # Phase 3
            # Phase 3: Train upsampling module
            params = list(self.model.upsampling_module.parameters())
            
        # Filter for parameters that require gradients
        trainable_params = [p for p in params if p.requires_grad]
        
        if not trainable_params:
            logger.warning(f"No trainable parameters found for phase {self.current_phase}")
            return None
            
        optimizer = torch.optim.Adam(trainable_params, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
    def on_train_epoch_start(self) -> None:
        """Handle phase transitions at epoch start."""
        current_epoch = self.current_epoch
        
        # Automatic phase progression (can be customized)
        if current_epoch == self.phase_1_epochs and self.current_phase == 1:
            self.set_training_phase(2)
            # Reconfigure optimizer for new phase
            self.trainer.optimizers = [self.configure_optimizers()]
            
        elif current_epoch == (self.phase_1_epochs + self.phase_2_epochs) and self.current_phase == 2:
            self.set_training_phase(3)
            # Reconfigure optimizer for new phase
            self.trainer.optimizers = [self.configure_optimizers()]
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Training step for 3DSR.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        outputs = self.model(batch)
        
        # Calculate loss based on training phase
        loss = self.loss_fn(
            outputs,
            batch,
            training_phase=self.current_phase,
        )
        
        # Log phase-specific losses
        if self.current_phase == 1:
            self.log("train_loss_phase1", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("vq_loss_top", outputs.get("vq_loss_top", 0.0), on_step=True, on_epoch=True)
            self.log("vq_loss_bottom", outputs.get("vq_loss_bottom", 0.0), on_step=True, on_epoch=True)
        elif self.current_phase == 2:
            self.log("train_loss_phase2", loss, on_step=True, on_epoch=True, prog_bar=True)
        else:
            self.log("train_loss_phase3", loss, on_step=True, on_epoch=True, prog_bar=True)
            
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("training_phase", float(self.current_phase), on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Validation step for 3DSR.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Batch with predictions
        """
        outputs = self.model(batch)
        batch.update(outputs)
        
        # Calculate validation loss
        val_loss = self.loss_fn(
            outputs,
            batch,
            training_phase=self.current_phase,
        )
        
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return batch
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Test step for 3DSR.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Batch with predictions
        """
        outputs = self.model(batch)
        batch.update(outputs)
        
        return batch
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Prediction step for 3DSR.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Batch with predictions
        """
        outputs = self.model(batch)
        batch.update(outputs)
        
        return batch
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        # Log current training phase info
        logger.info(f"Validation epoch {self.current_epoch} completed in phase {self.current_phase}")
        
    def on_train_epoch_end(self) -> None:
        """Called at the end of each training epoch."""
        # Log training progress
        logger.info(f"Training epoch {self.current_epoch} completed in phase {self.current_phase}")
        
        # You can add custom logic here for phase transitions based on validation metrics
        # instead of fixed epoch counts if needed