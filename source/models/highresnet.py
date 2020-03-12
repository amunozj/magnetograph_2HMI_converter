"""
Pytorch implementation of HighRes-net, a neural network for multi-frame super resolution (MFSR) by recursive fusion.
Github: https://github.com/ElementAI/HighRes-net
Paper: https://openreview.net/forum?id=HJxJ2h4tPr

"""

from source.models.model_utils import ResidualBlock, Encoder, Decoder

import torch
import torch.nn as nn


class FusionBlock(nn.Module):

    def __init__(self, input_channels=64, kernel_size=3):
        super(FusionBlock, self).__init__()

        self.fuse = nn.Sequential(ResidualBlock(2 * input_channels, kernel_size),
                                  nn.ReflectionPad2d(kernel_size // 2),
                                  nn.Conv2d(in_channels=2 * input_channels,
                                            out_channels=input_channels,
                                            kernel_size=kernel_size, padding=0),
                                  nn.PReLU())

    def forward(self, x):
        '''
        Fuse hidden states recursively.
        Args:
            x : tensor (B, L, C_h, H, W), hidden states, ordered by time
        Returns:
            out: tensor (B, C_h, H, W), fused hidden state
        '''

        batch_size, nviews, channels, heigth, width = x.shape
        parity = nviews % 2
        half_len = nviews // 2

        while half_len > 0:
            first_half = x[:, :half_len]  # first half hidden states (B, L/2, C_h, H, W)
            second_half = x[:, half_len:nviews - parity]  # second half hidden states (B, L/2, C_h, H, W)

            concat_state = torch.cat([first_half, second_half],
                                     2)  # concat hidden states accross channels (B, L/2, 2*C_h, H, W)
            concat_state = concat_state.view(-1, 2 * channels, heigth,
                                             width)  # reshape hidden states (B*L/2, 2*C_h, H, W)
            fused_state = self.fuse(
                concat_state)  # (B*L/2, C_h, H, W)      // Oldest frame (past) will be fused with LRt (future),  ....   , LRt (past) will be fused with latest frame (future)
            x = first_half + fused_state.view(batch_size, half_len, channels, heigth,
                                              width)  # new hidden states (B, L/2, C_h, H, W)

            nviews = half_len
            parity = nviews % 2
            half_len = nviews // 2

        return x[:, 0]  # (B, C_h, H, W)


class HighResNet(nn.Module):
    """
    HighRes_net, a neural network for multi-frame super resolution (MFSR) by recursive fusion.
    """

    def __init__(self, in_channels=1, enc_num_layers=2, kernel_size=3, hidden_channel_size=64,
                 upscale_factor=2, final_kernel_size=1):
        super().__init__()

        self.name = 'HighResNet_RPRC'
        self.upscale_factor = upscale_factor

        self.encode = Encoder(in_channels=2 * in_channels, num_layers=enc_num_layers,
                              kernel_size=kernel_size, channel_size=hidden_channel_size)
        self.fuse = FusionBlock(input_channels=hidden_channel_size, kernel_size=kernel_size)
        self.decode = Decoder(deconv_in_channels=hidden_channel_size,
                              deconv_out_channels=hidden_channel_size, upscale_factor=upscale_factor,
                              final_kernel_size=final_kernel_size)

    def forward(self, lrs):
        """
        Super resolve a batch of low-resolution images.
        Args:
            lrs : tensor (B, L, H, W) or (B, H, W), low-resolution images ordered by time stamp
        Returns:
            srs: tensor (B, H, W), super-resolved images
        Parameters
        ----------
        lrs

        Returns
        -------

        """
        if len(lrs.shape) == 3:  # (B, H, W)
            lrs = lrs[:, None]  # (B, L, H, W)

        batch_size, nviews, heigth, width = lrs.shape
        lrs = lrs.view(-1, nviews, 1, heigth, width)  # (B, L, C_in, H, W)

        # Using first layer (magnetogram) as reference image (LRt) aka anchor and position information as views
        # shared across multiple views (B, 1, C_in, H, W)
        refs = lrs[:, [0]]
        refs = refs.repeat(1, nviews - 1, 1, 1, 1)  # (B, L-1, C_in, H, W)
        stacked_input = torch.cat([lrs[:, 1:, :, :], refs], 2)  # (B, L-1, 2*C_in, H, W)
        stacked_input = stacked_input.view(batch_size * (nviews - 1), 2, heigth, width)  # (B*(L-1), 2*C_in, H, W)

        # embed inputs
        layer1 = self.encode(stacked_input)  # encode input tensor (B*(L-1), C_h, H, W)
        layer1 = layer1.view(batch_size, nviews - 1, -1, heigth, width)  # (B, L, C_h, H, W)

        # Duplicate LR_t (B, L+1, C_h, H, W)
        # layer1 = torch.cat([layer1[:,:nviews//2+1], layer1[:,nviews//2:]], 1)
        # or Remove LR_t (B, L-1, C_h, H, W)
        # layer1 = torch.cat([layer1[:,:nviews//2], layer1[:,nviews//2+1:]], 1)

        # fuse, upsample
        recursive_layer = self.fuse(layer1)  # fuse hidden states (B, C_h, H, W)
        srs = self.decode(recursive_layer)  # decode final hidden state (B, 3*H, 3*W)
        return srs
