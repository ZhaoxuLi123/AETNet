import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from models.swin import SwinTransformerBlock
from models.spectralTransformer import SpectralTransformerBlock
from timm.models.layers import DropPath, to_2tuple, trunc_normal_, lecun_normal_


class Transformer(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            st_heads,
            num_blocks,
            input_resolution,
            swin_heads
    ):
        super().__init__()
        window_size = input_resolution[0]//8
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                SpectralTransformerBlock(dim=dim, dim_head=dim_head, heads=st_heads),
                SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                     num_heads=swin_heads, window_size=window_size,
                                     shift_size=0),
                SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                     num_heads=swin_heads, window_size=window_size,
                                     shift_size=window_size//2),
                nn.Conv2d(dim*2, dim , 1, 1, bias=False),
            ]))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (spect, w_swin, sw_swin,_) in self.blocks:
            # x = spect(x)
            x = w_swin(x)
            x = sw_swin(x)
        out = x.permute(0, 3, 1, 2)
        return out


class AETNet(nn.Module):
    def __init__(self, in_dim=64, dim=64, image_size=64, stage=2, num_blocks=[1,1,1], pad = 4, use_unet =True, use_siwn =True):
        super(AETNet, self).__init__()
        self.dim = dim
        self.use_unet = use_unet
        self.use_siwn = use_siwn
        if self.use_unet :
            self.stage = stage
        else:
            self.stage = 0
        swin_heads_list =[2, 4, 8]
        self.pad = pad
        # Convolutional Encoder
        self.conv_encoder = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)
        # self.conv_encoder = nn.Conv2d(in_dim, self.dim, 1, 1, 0, bias=False)
        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        size_stage = image_size + 2 * self.pad
        assert size_stage % 8 == 0
        self.pad_add = 0
        for i in range(self.stage):
            self.encoder_layers.append(nn.ModuleList([
                Transformer(
                    dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, st_heads=dim_stage // dim, input_resolution=[size_stage,size_stage] ,
            swin_heads=swin_heads_list[i]),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2
            size_stage //= 2

        # Bottleneck
        self.bottleneck = Transformer(
            dim=dim_stage, dim_head=dim, st_heads=dim_stage // dim, num_blocks=num_blocks[-1], input_resolution=[size_stage,size_stage] ,
            swin_heads=swin_heads_list[-1])
        # Decoder

        self.decoder_layers = nn.ModuleList([])
        for i in range(self.stage):
            size_stage *= 2
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                Transformer(
                    dim=dim_stage // 2,  num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                    st_heads= (dim_stage // 2) // dim, input_resolution=[size_stage,size_stage] ,
            swin_heads= swin_heads_list[stage - 1 - i]),#5* image_size // size_stage
            ]))
            dim_stage //= 2

        # Convolutional Decoder
        self.conv_decoder = nn.Conv2d(self.dim, in_dim, 3, 1, 1, bias=False)
        # self.conv_decoder = nn.Conv2d(self.dim, in_dim, 1, 1, 0, bias=False)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        # Padding
        b, c, h, w = x.shape
        h_add = math.ceil((h + self.pad*2)/72)*72 - h - self.pad*2
        h_add1 = math.floor(h_add/2)
        h_add2 = h_add-h_add1
        w_add = math.ceil((w + self.pad*2)/72)*72 - w - self.pad*2
        w_add1 = math.floor(w_add/2)
        w_add2 = w_add-w_add1
        x = F.pad(x, [self.pad+h_add1, self.pad+h_add2, self.pad+w_add1, self.pad+w_add2], mode='reflect')
        # Convolutional Encoder
        fea = self.conv_encoder(x)
        # Swin Unet Encoder
        fea_encoder = []
        for (SwinBlock, FeaDownSample) in self.encoder_layers:
            if self.use_siwn:
                fea = SwinBlock(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Swin Unet Bottleneck
        if self.use_unet and self.use_siwn:
            fea = self.bottleneck(fea)

        #  Swin Unet Decoder
        for i, (FeaUpSample, Fusion, SwinBlock) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fusion(torch.cat([fea, fea_encoder[self.stage-1-i]], dim=1))
            if self.use_siwn:
                fea = SwinBlock(fea)
        # Convolutional Decoder
        res = self.conv_decoder(fea)
        out = res + x
        out = out[:, :, (self.pad+h_add1):-(self.pad+h_add2), (self.pad+w_add1):-(self.pad+w_add2)]
        return out