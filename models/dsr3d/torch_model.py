"""3DSR PyTorch Model Implementation."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
import pickle
from pathlib import Path

__all__ = ["Dsr3dModel"]


# =============================================
# Vector Quantization Components
# =============================================

class VectorQuantizerEMA(nn.Module):
    """EMA-based Vector Quantizer from 3DSR."""
    
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def get_quantized(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        quantized = inputs + (quantized - inputs).detach()

        return quantized.permute(0, 3, 1, 2).contiguous()

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


# =============================================
# Residual Components
# =============================================

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens, groups=1):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(False),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False, groups=groups),
            nn.ReLU(False),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False, groups=groups)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, groups=1):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens, groups=groups)
                                      for _ in range(self._num_residual_layers)])
        self.relu = nn.ReLU(False)

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return self.relu(x)


# =============================================
# Encoder/Decoder Components
# =============================================

class EncoderBot(nn.Module):
    """Bottom encoder for 3DSR with depth+RGB support."""
    
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(EncoderBot, self).__init__()
        self.input_conv_depth = nn.Conv2d(in_channels=1,
                                         out_channels=num_hiddens//4,
                                         kernel_size=1,
                                         stride=1, padding=0)
        self.input_conv_rgb = nn.Conv2d(in_channels=in_channels-1,
                                       out_channels=num_hiddens//4,
                                       kernel_size=1,
                                       stride=1, padding=0)

        self._conv_1 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2, padding=1, groups=2)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1, groups=2)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1, groups=2)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens, groups=2)

        self.relu = nn.ReLU(False)

    def forward(self, inputs):
        x_d = self.input_conv_depth(inputs[:,:1,:,:])
        x_rgb = self.input_conv_rgb(inputs[:,1:,:,:])
        x = torch.cat((x_d, x_rgb), dim=1)
        x = self.relu(x)
        x = self._conv_1(x)
        x = self.relu(x)

        x = self._conv_2(x)
        x = self.relu(x)

        x = self._conv_3(x)
        return self._residual_stack(x)


class EncoderTop(nn.Module):
    """Top encoder for 3DSR."""
    
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(EncoderTop, self).__init__()
        self._conv_1 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1, groups=2)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1, groups=2)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens, groups=2)

        self.relu = nn.ReLU(False)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self.relu(x)

        x = self._conv_2(x)
        x = self.relu(x)

        x = self._residual_stack(x)
        return x


class DecoderBot(nn.Module):
    """Bottom decoder for 3DSR with separate depth and RGB outputs."""
    
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, out_channels=4):
        super(DecoderBot, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1, groups=2)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens, groups=2)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4,
                                                stride=2, padding=1, groups=2)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
                                                out_channels=num_hiddens//4,
                                                kernel_size=4,
                                                stride=2, padding=1, groups=2)
        self._conv_out_depth = nn.Conv2d(in_channels=num_hiddens//8,
                                         out_channels=1,
                                         kernel_size=1,
                                         stride=1, padding=0, groups=1)
        self._conv_out_rgb = nn.Conv2d(in_channels=num_hiddens//8,
                                       out_channels=3,
                                       kernel_size=1,
                                       stride=1, padding=0, groups=1)

        self.relu = nn.ReLU(False)

    def forward(self, inputs):
        b, c, h, w = inputs.shape
        in_t = inputs[:,:c//2,:,:]
        in_b = inputs[:,c//2:,:,:]

        b, c, h, w = in_t.shape
        in_t_d = in_t[:,:c//2,:,:]
        in_t_rgb = in_t[:,c//2:,:,:]
        in_b_d = in_b[:,:c//2,:,:]
        in_b_rgb = in_b[:,c//2:,:,:]
        in_joined = torch.cat((in_t_d, in_b_d, in_t_rgb, in_b_rgb), dim=1)
        x = self._conv_1(in_joined)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = self.relu(x)
        x = self._conv_trans_2(x)
        c = x.shape[1]
        x_d = x[:,:c//2,:,:]
        x_rgb = x[:,c//2:,:,:]
        out_d = self._conv_out_depth(x_d)
        out_rgb = self._conv_out_rgb(x_rgb)
        x_out = torch.cat((out_d, out_rgb), dim=1)

        return x_out


class PreVQBot(nn.Module):
    """Pre-quantization processing for bottom features."""
    
    def __init__(self, num_hiddens, embedding_dim):
        super(PreVQBot, self).__init__()
        self.num_hiddens = num_hiddens
        self.embedding_dim = embedding_dim
        self._pre_vq_conv_bot = nn.Conv2d(in_channels=num_hiddens + embedding_dim,
                                          out_channels=embedding_dim,
                                          kernel_size=1,
                                          stride=1, groups=2)

    def forward(self, inputs):
        in_enc_b = inputs[:,:self.num_hiddens,:,:]
        in_up_t = inputs[:,self.num_hiddens:,:,:]

        in_enc_b_d = in_enc_b[:,:self.num_hiddens//2,:,:]
        in_enc_b_rgb = in_enc_b[:,self.num_hiddens//2:,:,:]
        in_up_t_d = in_up_t[:,:self.embedding_dim//2,:,:]
        in_up_t_rgb = in_up_t[:,self.embedding_dim//2:,:,:]

        in_joined = torch.cat((in_enc_b_d, in_up_t_d, in_enc_b_rgb, in_up_t_rgb), dim=1)
        x = self._pre_vq_conv_bot(in_joined)

        return x


# =============================================
# DSR Components (for anomaly detection)
# =============================================

class FeatureEncoder(nn.Module):
    def __init__(self, in_channels, base_width):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width),
            nn.ReLU(inplace=True))
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 2),
            nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 4),
            nn.ReLU(inplace=True))

    def forward(self, x):
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2)
        return b1, b2, b3


class FeatureDecoder(nn.Module):
    def __init__(self, base_width, out_channels=1):
        super().__init__()

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
                                 nn.InstanceNorm2d(base_width * 2),
                                 nn.ReLU(inplace=True))

        self.db2 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 2),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
                                 nn.InstanceNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))

    def forward(self, b1, b2, b3):
        up2 = self.up2(b3)
        db2 = self.db2(up2)

        up3 = self.up3(db2)
        db3 = self.db3(up3)

        out = self.fin_out(db3)
        return out


class SubspaceRestrictionNetwork(nn.Module):
    """Subspace restriction network from DSR."""
    
    def __init__(self, in_channels=64, out_channels=64, base_width=64):
        super().__init__()
        self.base_width = base_width
        self.encoder = FeatureEncoder(in_channels, self.base_width)
        self.decoder = FeatureDecoder(self.base_width, out_channels=out_channels)

    def forward(self, x):
        b1, b2, b3 = self.encoder(x)
        output = self.decoder(b1, b2, b3)
        return output


class UnetEncoder(nn.Module):
    def __init__(self, in_channels, base_width):
        super().__init__()
        norm_layer = nn.InstanceNorm2d
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            norm_layer(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            norm_layer(base_width),
            nn.ReLU(inplace=True))
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            norm_layer(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            norm_layer(base_width * 2),
            nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True))
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True))

    def forward(self, x):
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        return b1, b2, b3, b4


class UnetDecoder(nn.Module):
    def __init__(self, base_width, out_channels=1):
        super().__init__()
        norm_layer = nn.InstanceNorm2d
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
                                 norm_layer(base_width * 4),
                                 nn.ReLU(inplace=True))
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width * (4 + 4), base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
                                 norm_layer(base_width * 2),
                                 nn.ReLU(inplace=True))
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width * (2 + 2), base_width * 2, kernel_size=3, padding=1),
            norm_layer(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            norm_layer(base_width * 2),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
                                 norm_layer(base_width),
                                 nn.ReLU(inplace=True))
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width * (1 + 1), base_width, kernel_size=3, padding=1),
            norm_layer(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            norm_layer(base_width),
            nn.ReLU(inplace=True)
        )

        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))

    def forward(self, b1, b2, b3, b4):
        up1 = self.up1(b4)
        cat1 = torch.cat((up1, b3), dim=1)
        db1 = self.db1(cat1)

        up2 = self.up2(db1)
        cat2 = torch.cat((up2, b2), dim=1)
        db2 = self.db2(cat2)

        up3 = self.up3(db2)
        cat3 = torch.cat((up3, b1), dim=1)
        db3 = self.db3(cat3)

        out = self.fin_out(db3)
        return out


class UnetModel(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, base_width=64):
        super().__init__()
        self.encoder = UnetEncoder(in_channels, base_width)
        self.decoder = UnetDecoder(base_width, out_channels=out_channels)

    def forward(self, x):
        b1, b2, b3, b4 = self.encoder(x)
        output = self.decoder(b1, b2, b3, b4)
        return output


class AnomalyDetectionModule(nn.Module):
    """Anomaly detection module from DSR."""
    
    def __init__(self, in_channels=8, base_width=32):
        super(AnomalyDetectionModule, self).__init__()
        self.unet = UnetModel(in_channels=in_channels, out_channels=2, base_width=base_width)

    def forward(self, image_real, image_anomaly):
        img_x = torch.cat((image_real, image_anomaly), dim=1)
        x = self.unet(img_x)
        return x


# =============================================
# Main 3DSR Model
# =============================================

class DiscreteLatentModelGroups(nn.Module):
    """3DSR Discrete Latent Model with depth+RGB support."""
    
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim,
                 commitment_cost, decay=0, test=False, in_channels=4, out_channels=4):
        super(DiscreteLatentModelGroups, self).__init__()
        self.test = test
        self._encoder_t = EncoderTop(num_hiddens, num_hiddens,
                                     num_residual_layers,
                                     num_residual_hiddens)

        self._encoder_b = EncoderBot(in_channels, num_hiddens,
                                     num_residual_layers,
                                     num_residual_hiddens)

        self._pre_vq_conv_top = nn.Conv2d(in_channels=num_hiddens,
                                          out_channels=embedding_dim,
                                          kernel_size=1,
                                          stride=1, groups=2)

        self._pre_vq_conv_bot = PreVQBot(num_hiddens, embedding_dim)

        self._vq_vae_top = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                              commitment_cost, decay)

        self._vq_vae_bot = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                              commitment_cost, decay)

        self._decoder_b = DecoderBot(embedding_dim*2,
                                     num_hiddens,
                                     num_residual_layers,
                                     num_residual_hiddens, out_channels=out_channels)

        self.upsample_t = nn.ConvTranspose2d(
            embedding_dim, embedding_dim, 4, stride=2, padding=1, groups=2
        )

    def forward(self, x):
        #Encoder Hi
        enc_b = self._encoder_b(x)

        #Encoder Lo -- F_Lo
        enc_t = self._encoder_t(enc_b)
        zt = self._pre_vq_conv_top(enc_t)

        # Quantize F_Lo with K_Lo
        loss_t, quantized_t, perplexity_t, encodings_t = self._vq_vae_top(zt)
        # Upsample Q_Lo
        up_quantized_t = self.upsample_t(quantized_t)

        # Concatenate and transform the output of Encoder_Hi and upsampled Q_lo -- F_Hi
        feat = torch.cat((enc_b, up_quantized_t), dim=1)
        zb = self._pre_vq_conv_bot(feat)

        # Quantize F_Hi with K_Hi
        loss_b, quantized_b, perplexity_b, encodings_b = self._vq_vae_bot(zb)

        # Concatenate Q_Hi and Q_Lo and input it into the General appearance decoder
        quant_join = torch.cat((up_quantized_t, quantized_b), dim=1)
        recon_fin = self._decoder_b(quant_join)

        return loss_b, loss_t, recon_fin, quantized_t, quantized_b


class Dsr3dModel(nn.Module):
    """3DSR Model for 3D surface anomaly detection.
    
    This model extends the 2D DSR approach to work with RGB+Depth data
    for enhanced 3D surface anomaly detection.
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
    ):
        super().__init__()
        
        self.rgb_channels = rgb_channels
        self.depth_channels = depth_channels
        self.use_depth_only = use_depth_only
        self.latent_anomaly_strength = latent_anomaly_strength
        self.upsampling_train_ratio = upsampling_train_ratio
        
        # Input channels depend on whether we use depth only or RGB+Depth
        input_channels = depth_channels if use_depth_only else rgb_channels + depth_channels
        
        # Initialize the discrete latent model (VQ-VAE part)
        self.discrete_model = DiscreteLatentModelGroups(
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            decay=decay,
            in_channels=input_channels,
            out_channels=input_channels
        )
        
        # Object-specific decoder (DSR subspace restriction)
        self.object_specific_decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            SubspaceRestrictionNetwork(
                in_channels=embedding_dim,
                out_channels=input_channels,
                base_width=embedding_dim//2
            )
        )
        
        # Anomaly detection module
        self.anomaly_detection_module = AnomalyDetectionModule(
            in_channels=input_channels * 2,  # Concatenated real + reconstructed
            base_width=32
        )
        
        # Upsampling module for final anomaly maps
        self.upsampling_module = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Load pretrained VQ model if provided
        if pretrained_vq_model_path and Path(pretrained_vq_model_path).exists():
            self.load_pretrained_vq_model(pretrained_vq_model_path)
            
        # Training phase (1: VQ+reconstruction, 2: anomaly detection, 3: upsampling)
        self.training_phase = 1
        
    def load_pretrained_vq_model(self, model_path: str) -> None:
        """Load pretrained vector quantization model."""
        try:
            with open(model_path, 'rb') as f:
                pretrained_weights = pickle.load(f)
            
            # Load the pretrained weights into the discrete model
            if hasattr(pretrained_weights, 'state_dict'):
                self.discrete_model.load_state_dict(pretrained_weights.state_dict(), strict=False)
            else:
                print(f"Loaded pretrained VQ model from {model_path}")
                
        except Exception as e:
            print(f"Warning: Could not load pretrained VQ model from {model_path}: {e}")
    
    def set_training_phase(self, phase: int) -> None:
        """Set the current training phase."""
        assert phase in [1, 2, 3], "Training phase must be 1, 2, or 3"
        self.training_phase = phase
        
        # Freeze/unfreeze different parts based on training phase
        if phase == 1:
            # Phase 1: Train VQ model and general reconstruction
            for param in self.discrete_model.parameters():
                param.requires_grad = True
            for param in self.object_specific_decoder.parameters():
                param.requires_grad = True
            for param in self.anomaly_detection_module.parameters():
                param.requires_grad = False
            for param in self.upsampling_module.parameters():
                param.requires_grad = False
                
        elif phase == 2:
            # Phase 2: Train anomaly detection module
            for param in self.discrete_model.parameters():
                param.requires_grad = False
            for param in self.object_specific_decoder.parameters():
                param.requires_grad = False
            for param in self.anomaly_detection_module.parameters():
                param.requires_grad = True
            for param in self.upsampling_module.parameters():
                param.requires_grad = False
                
        else:  # phase == 3
            # Phase 3: Train upsampling module
            for param in self.discrete_model.parameters():
                param.requires_grad = False
            for param in self.object_specific_decoder.parameters():
                param.requires_grad = False
            for param in self.anomaly_detection_module.parameters():
                param.requires_grad = False
            for param in self.upsampling_module.parameters():
                param.requires_grad = True
    
    def generate_synthetic_anomalies(self, quantized_features: torch.Tensor) -> torch.Tensor:
        """Generate synthetic anomalies in the latent space."""
        if not self.training:
            return quantized_features
            
        # Add noise to create synthetic anomalies
        noise = torch.randn_like(quantized_features) * self.latent_anomaly_strength
        synthetic_anomalies = quantized_features + noise
        
        return synthetic_anomalies
    
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        """Forward pass of 3DSR model."""
        # Handle both dict and anomalib Batch objects
        if hasattr(batch, 'image'):
            # Anomalib Batch object
            rgb_data = batch.image
            depth_data = getattr(batch, 'depth', None)
        else:
            # Dictionary format
            rgb_data = batch["image"]
            depth_data = batch.get("depth", None)
        
        # Extract and prepare input data
        if self.use_depth_only:
            # Use only depth information
            if depth_data is not None:
                input_data = depth_data
            else:
                # Fallback: use first channel of image as depth
                input_data = rgb_data[:, :1]
        else:
            # Concatenate RGB and depth
            if depth_data is not None:
                # Depth + RGB concatenation
                input_data = torch.cat([depth_data, rgb_data], dim=1)
            else:
                # If no depth channel, handle different RGB formats
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
        
        # Phase 1: VQ-VAE processing
        loss_b, loss_t, general_reconstruction, quantized_t, quantized_b = self.discrete_model(input_data)
        
        # Object-specific reconstruction through subspace restriction
        object_specific_reconstruction = self.object_specific_decoder(quantized_b)
        
        # Ensure all reconstructions match input size
        target_size = input_data.shape[-2:]
        
        if general_reconstruction.shape[-2:] != target_size:
            general_reconstruction = F.interpolate(
                general_reconstruction, size=target_size, mode='bilinear', align_corners=False
            )
            
        if object_specific_reconstruction.shape[-2:] != target_size:
            object_specific_reconstruction = F.interpolate(
                object_specific_reconstruction, size=target_size, mode='bilinear', align_corners=False
            )
        
        outputs = {
            "general_reconstruction": general_reconstruction,
            "object_specific_reconstruction": object_specific_reconstruction,
            "vq_loss_top": loss_t,
            "vq_loss_bottom": loss_b,
            "quantized_top": quantized_t,
            "quantized_bottom": quantized_b,
        }
        
        # Phase 2 & 3: Anomaly detection
        if self.training_phase >= 2:
            # Generate synthetic anomalies for training
            if self.training:
                synthetic_quantized = self.generate_synthetic_anomalies(quantized_b)
                synthetic_reconstruction = self.object_specific_decoder(synthetic_quantized)
                
                # Ensure size matching
                if synthetic_reconstruction.shape[-2:] != target_size:
                    synthetic_reconstruction = F.interpolate(
                        synthetic_reconstruction, size=target_size, mode='bilinear', align_corners=False
                    )
                
                # Anomaly detection on synthetic data
                anomaly_logits = self.anomaly_detection_module(
                    object_specific_reconstruction, 
                    synthetic_reconstruction
                )
            else:
                # During inference, compare with general reconstruction
                anomaly_logits = self.anomaly_detection_module(
                    object_specific_reconstruction,
                    general_reconstruction
                )
            
            outputs["anomaly_logits"] = anomaly_logits
            
            # Phase 3: Upsampling for final anomaly map
            if self.training_phase >= 3:
                anomaly_map = self.upsampling_module(anomaly_logits)
                outputs["anomaly_map"] = anomaly_map
                outputs["pred_score"] = torch.max(anomaly_map.view(anomaly_map.size(0), -1), dim=1)[0]
            else:
                # Use logits directly as anomaly map for phase 2
                anomaly_map = torch.sigmoid(anomaly_logits[:, 1:2])  # Take anomaly channel
                outputs["anomaly_map"] = anomaly_map
                outputs["pred_score"] = torch.max(anomaly_map.view(anomaly_map.size(0), -1), dim=1)[0]
        else:
            # Phase 1: No anomaly detection yet, use reconstruction error
            if object_specific_reconstruction.shape[-2:] != input_data.shape[-2:]:
                object_specific_reconstruction_resized = F.interpolate(
                    object_specific_reconstruction,
                    size=input_data.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            else:
                object_specific_reconstruction_resized = object_specific_reconstruction
                
            reconstruction_error = F.mse_loss(
                object_specific_reconstruction_resized, 
                input_data, 
                reduction='none'
            ).mean(dim=1, keepdim=True)
            
            outputs["anomaly_map"] = reconstruction_error
            outputs["pred_score"] = torch.max(reconstruction_error.view(reconstruction_error.size(0), -1), dim=1)[0]
        
        return outputs