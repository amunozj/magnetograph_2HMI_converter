""" 
Util modules to build neural networks in pytorch with reflection padding
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, channel_size=64, kernel_size=3, p=0.0):

        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(nn.ReflectionPad2d(kernel_size//2),
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=0),
            nn.PReLU(),
            nn.ReflectionPad2d(kernel_size // 2),
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=0),
            nn.Dropout(p=p),
            nn.PReLU()
        )

    def forward(self, x):
        '''
        Args:
            x : tensor (B, C, W, H), hidden state
        Returns:
            x + residual: tensor (B, C, W, H), new hidden state
        '''

        residual = self.block(x)
        return x + residual


class Encoder(nn.Module):

    def __init__(self, in_channels=2, num_layers=2, kernel_size=3, channel_size=64, p=0.0):

        super(Encoder, self).__init__()

        self.init_layer = nn.Sequential(nn.ReflectionPad2d(kernel_size//2),
            nn.Conv2d(in_channels=in_channels, out_channels=channel_size, kernel_size=kernel_size, padding=0),
            nn.Dropout(p=p),
            nn.PReLU())

        res_layers = [ResidualBlock(channel_size, kernel_size) for _ in range(num_layers)]
        self.res_layers = nn.Sequential(*res_layers)

        self.final = nn.Sequential(nn.ReflectionPad2d(kernel_size//2),
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=0)
        )

    def forward(self, x):
        '''
        Encodes an input tensor x.
        Args:
            x : tensor (B, C_in, H, W), input images
        Returns:
            out: tensor (B, C, H, W), hidden states
        '''

        x = self.init_layer(x) # (B, C_h, H, W)
        x = self.res_layers(x) # (B, C_h, H, W)
        x = self.final(x) # (B, C_h, H, W)
        return x


class TemporalSmoothing(nn.Module):

    def __init__(self, input_channels=64, kernel_size=3, p=0.0):

        super(TemporalSmoothing, self).__init__()

        self.fuse = nn.Sequential(ResidualBlock(2*input_channels, kernel_size),
                                nn.ReflectionPad2d(kernel_size // 2),
                                nn.Conv2d(in_channels=2*input_channels, out_channels=input_channels, kernel_size=kernel_size, padding=0),
                                nn.Dropout(p=p),
                                nn.PReLU())

    def forward(self, x):
        '''
        Fuse two hidden states: s_t+ and s_t-
        Args:
            x : tensor (B, 2, C_h, H, W), pair of hidden states
        Returns:
            out: tensor (B, C_h, H, W), fused hidden state
        '''

        batch_size, nviews, channels, heigth, width = x.shape # (B, 2, C_h, H, W)
        x = x.view(batch_size, nviews*channels, heigth, width) # reshape hidden states (B, 2*C_h, H, W)
        x = self.fuse(x) # fused state (B, C_h, H, W)
        return x


class Decoder(nn.Module):

    def __init__(self, deconv_in_channels=64, deconv_out_channels=64, upscale_factor=3, final_kernel_size=1, p=0.0):

        super(Decoder, self).__init__()

        self.deconv = nn.Sequential(nn.Upsample(scale_factor=upscale_factor, mode='bilinear'),
                                     nn.ReflectionPad2d(1),
                                     nn.Conv2d(in_channels=deconv_in_channels,
                                               out_channels=deconv_out_channels,
                                               kernel_size=3, stride=1, padding=0),
                                     nn.Dropout(p=p),
                                     nn.PReLU())

        self.final = nn.Sequential(nn.ReflectionPad2d(final_kernel_size// 2),
                      nn.Conv2d(in_channels=deconv_out_channels,
                               out_channels=1,
                               kernel_size=final_kernel_size,
                               padding=0))

    def forward(self, x):
        '''
        Decode a hidden state x.
        Args:
            x : tensor (B, C_h, W, H), fused hidden state
        Returns:
            out: tensor (B, upscale_factor*W, upscale_factor*H), super-resolved image
        '''

        x = self.deconv(x) # (B, C_h, upscale_factor*W, upscale_factor*H)
        x = self.final(x) # (B, 1, upscale_factor*W, upscale_factor*H)
        return x[:,0]
