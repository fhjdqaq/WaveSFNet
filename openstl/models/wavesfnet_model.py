import torch
import torch.nn as nn

from openstl.modules.wavesf_modules import Encoder, Decoder, MidSTBNet

class WaveSF_Model(nn.Module):

    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4,
                 mlp_ratio=4., k_spatial=9,
                 drop=0.0, drop_path=0.0,
                 spatio_kernel_enc=3, spatio_kernel_dec=3,
                 act_inplace=True,
                 **kwargs):
        super().__init__()
        T, C, H, W = in_shape

        """H_lat = int(H / 2 ** (N_S / 2))
        W_lat = int(W / 2 ** (N_S / 2))"""
        scale = 2 ** N_S
        assert H % scale == 0 and W % scale == 0, f"H,W must be divisible by 2**N_S, got {(H,W)} and N_S={N_S}"
        H_lat = H // scale
        W_lat = W // scale
 

        # Encoder / Decoder
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.dec = Decoder(hid_S, C, N_S, spatio_kernel_dec, act_inplace=act_inplace)

        # Translator
        self.hid = MidSTBNet(
            channel_in=T * hid_S,
            channel_hid=hid_T,
            N2=N_T,
            T=T,  
            mlp_ratio=mlp_ratio,
            k_spatial=k_spatial,
            drop=drop,
            drop_path=drop_path,
            resolution=(H_lat, W_lat)
        )

        self.T = T

    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B * T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)

        hid = hid.reshape(B * T, C_, H_, W_)
        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)
        return Y