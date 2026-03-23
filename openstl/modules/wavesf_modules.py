import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_
from .simvp_modules import ConvSC

class TemporalDifferenceInjector(nn.Module):
    def __init__(self, dim, num_frames):
        super().__init__()
        self.dim = dim
        self.num_frames = num_frames
        
        self.motion_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.SiLU()
        )
        self.motion_gate = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        # x: [B, TC, H, W]
        B, TC, H, W = x.shape
        T = self.num_frames

        assert TC % T == 0, \
            f"TemporalDifferenceInjector expects TC divisible by T, got TC={TC}, T={T}"

        C_per_frame = TC // T
        x_seq = x.view(B, T, C_per_frame, H, W)

        x_prev = torch.cat([x_seq[:, :1], x_seq[:, :-1]], dim=1)
        x_diff = x_seq - x_prev

        x_diff_flat = x_diff.view(B, TC, H, W)
        motion_feat = self.motion_conv(x_diff_flat)

        return x + self.motion_gate * motion_feat



class GRN(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.eps = eps

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * Nx) + self.beta + x

class SpectralGating(nn.Module):
    def __init__(self, dim, h, w):
        super().__init__()
        self.dim = dim
        self.freq_weight = nn.Parameter(
            torch.randn(dim, h, w // 2 + 1, 2, dtype=torch.float32) * 0.02
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_fft = torch.fft.rfft2(x, norm='ortho')
        weight = torch.view_as_complex(self.freq_weight)
        x_fft = x_fft * weight
        x = torch.fft.irfft2(x_fft, s=(H, W), norm='ortho')
        return x

#  STBSubBlock 

class STBSubBlock(nn.Module):
    def __init__(self, dim, k_spatial=9, mlp_ratio=4.,
                 drop=0., drop_path=0.1, init_value=1e-2,
                 resolution=(64, 64), num_frames=4):
        super().__init__()
        
        def get_valid_groups(d, target=32):
            for g in range(target, 0, -1):
                if d % g == 0:
                    return g
            return 1 # Fallback to LayerNorm behavior

        valid_groups = get_valid_groups(dim, 32)
        # ----------------------------------------

        # 2. Token Mixing
        self.norm_tm = nn.GroupNorm(num_groups=valid_groups, num_channels=dim)
        pad = k_spatial // 2
        self.dw_spatial = nn.Conv2d(dim, dim, kernel_size=k_spatial, padding=pad, groups=dim, bias=False)
        self.spectral_gate = SpectralGating(dim, resolution[0], resolution[1])
        self.fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.alpha_fuse = nn.Parameter(torch.zeros(1))
        self.act_tm = nn.GELU()
        self.grn_tm = GRN(dim)
        self.gamma_tm = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)

        # 3. Channel Mixing
        self.norm_cm = nn.GroupNorm(num_groups=valid_groups, num_channels=dim)
        hidden_dim = int(dim * mlp_ratio)
        hidden_dim = max(hidden_dim, dim)

        self.pw1 = nn.Conv2d(dim, 2 * hidden_dim, 1, bias=True)
        self.dw_mid = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False)
        self.pw2 = nn.Conv2d(hidden_dim, dim, 1, bias=True)
        self.act_glu = nn.SiLU(inplace=False)

        self.grn_cm = GRN(dim)
        self.gamma_cm = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)
        self.drop = nn.Dropout(drop) if drop > 0. else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: [B, TC, H, W]

        shortcut = x
        x_norm = self.norm_tm(x)
        x_spa = self.dw_spatial(x_norm)
        x_spec = self.spectral_gate(x_norm)
        x_ctx = x_spa + x_spec
        x_ctx = x_ctx + self.alpha_fuse * self.fuse(x_ctx)

        y = self.act_tm(x_ctx)
        y = y.permute(0, 2, 3, 1)
        y = self.grn_tm(y)
        y = y.permute(0, 3, 1, 2)
        y = self.gamma_tm.view(1, -1, 1, 1) * y
        y = self.drop(y)
        x = shortcut + self.drop_path(y)

        # CM
        shortcut = x
        z = self.norm_cm(x)

        uv = self.pw1(z)
        u, v = torch.chunk(uv, 2, dim=1)
        v = self.dw_mid(v)
        z = self.act_glu(u) * v
        z = self.pw2(z)


        z = z.permute(0, 2, 3, 1)
        z = self.grn_cm(z)
        z = z.permute(0, 3, 1, 2)
        z = self.gamma_cm.view(1, -1, 1, 1) * z
        z = self.drop(z)
        x = shortcut + self.drop_path(z)
        return x

#  MidSTBNet: Time-Distributed Projection

class MidSTBNet(nn.Module):
    def __init__(self, channel_in, channel_hid, N2, T, 
                 mlp_ratio=2., k_spatial=9,
                 drop=0.0, drop_path=0.1,
                 resolution=(64, 64)):
        super().__init__()


        self.channel_in = channel_in
        self.channel_hid = channel_hid
        self.T = T
        
        if channel_hid % T != 0:
            self.pad_hid = T - (channel_hid % T)
            self.real_hid = channel_hid + self.pad_hid
        else:
            self.pad_hid = 0
            self.real_hid = channel_hid
            
        self.proj_in = nn.Conv2d(channel_in, self.real_hid, 1, groups=T, bias=True)
        self.tdm = TemporalDifferenceInjector(self.real_hid, T)

        self.proj_out = nn.Conv2d(self.real_hid, channel_in, 1, groups=T, bias=True)

        if N2 > 1 and drop_path > 0:
            dpr = [x.item() for x in torch.linspace(1e-2, drop_path, N2)]
        else:
            dpr = [drop_path] * N2

        blocks = []
        for i in range(N2):
            blocks.append(
                STBSubBlock(
                    dim=self.real_hid,
                    k_spatial=k_spatial,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=(drop_path if i == N2 - 1 else dpr[i]),
                    init_value=1e-2,
                    resolution=resolution,
                    num_frames=T
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B, T*C, H, W)
        
        # Frame-wise Projection
        x_embed = self.proj_in(x) 

        x_embed = self.tdm(x_embed)
        
        x_out = self.blocks(x_embed)
        
        # Frame-wise Back-Projection
        x_back = self.proj_out(x_out)
        
        x_back = x_back.view(B, T, C, H, W)
        return x_back


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def idwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = \
        in_batch, int(in_channel // (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    h = torch.zeros([out_batch, out_channel, out_height, out_width], device=x.device).float()
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h

class DWT(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return dwt_init(x)

class IDWT(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return idwt_init(x)

class FreqSelectBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=2., drop_path=0.):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        
        self.mlp_attn = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.SiLU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )
        
        self.conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        
        self.norm2 = nn.GroupNorm(1, dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * 1e-2)

    def forward(self, x):
        shortcut = x
        x_norm = self.norm1(x)
        
        x_avg = F.adaptive_avg_pool2d(x_norm, 1)
        x_max = F.adaptive_max_pool2d(x_norm, 1)
        
        attn = self.mlp_attn(x_avg + x_max)
        
        x_feat = x_norm * attn
        x_feat = self.conv(x_feat)
        
        x = shortcut + self.drop_path(self.layer_scale * x_feat)
        x = x + self.drop_path(self.layer_scale * self.mlp(self.norm2(x)))
        return x

class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S, spatio_kernel=None, act_inplace=True):
        super().__init__()
        self.stages = nn.ModuleList()
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_hid // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(C_hid // 2, C_hid // 2, kernel_size=3, padding=1),
        )
        curr_dim = C_hid // 2
        num_downsamples = N_S
        for i in range(num_downsamples):
            next_dim = C_hid if i == num_downsamples - 1 else curr_dim * 2
            layers = []
            layers.append(DWT()) 
            layers.append(nn.Conv2d(curr_dim * 4, next_dim, 1))
            layers.append(FreqSelectBlock(next_dim))
            layers.append(FreqSelectBlock(next_dim))
            self.stages.append(nn.Sequential(*layers))
            curr_dim = next_dim
        self.final_proj = nn.Identity()

    def forward(self, x):
        x = self.stem(x)
        skip_connection = x 
        for stage in self.stages:
            x = stage(x)
        return x, skip_connection

class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S, spatio_kernel=None, act_inplace=True):
        super().__init__()
        self.stages = nn.ModuleList()
        num_upsamples = N_S
        curr_dim = C_hid
        for i in range(num_upsamples):
            next_dim = curr_dim // 2 if i < num_upsamples - 1 else C_hid // 2
            layers = []
            layers.append(FreqSelectBlock(curr_dim))
            layers.append(nn.Conv2d(curr_dim, next_dim * 4, 1))
            layers.append(IDWT())
            self.stages.append(nn.Sequential(*layers))
            curr_dim = next_dim
        self.readout = nn.Sequential(
            nn.Conv2d(curr_dim, C_out, kernel_size=3, padding=1),
        )

    def forward(self, hid, enc1=None):
        x = hid
        for i, stage in enumerate(self.stages):
            x = stage(x)
            
            if i == len(self.stages) - 1 and enc1 is not None:
                x = x + enc1
        
        Y = self.readout(x)
        return Y
"""


def sampling_generator(N_S: int, reverse: bool = False):
    assert N_S >= 1, "at least 1"
    samplings = [False, True] * N_S  # length = 2*N_S, downsamples = N_S
    return list(reversed(samplings)) if reverse else samplings


class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S, spatio_kernel=None, act_inplace=True):
        super().__init__()
        if spatio_kernel is None:
            spatio_kernel = 3
        samplings = sampling_generator(N_S)  # len = 2*N_S

        layers = [ConvSC(C_in, C_hid, spatio_kernel,
                         downsampling=samplings[0],
                         act_inplace=act_inplace)]
        for s in samplings[1:]:
            layers.append(ConvSC(C_hid, C_hid, spatio_kernel,
                                 downsampling=s,
                                 act_inplace=act_inplace))
        self.enc = nn.Sequential(*layers)

    def forward(self, x):
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S, spatio_kernel=None, act_inplace=True):
        super().__init__()
        if spatio_kernel is None:
            spatio_kernel = 3
        samplings = sampling_generator(N_S, reverse=True)  # len = 2*N_S

        layers = []
        for s in samplings[:-1]:
            layers.append(ConvSC(C_hid, C_hid, spatio_kernel,
                                 upsampling=s,
                                 act_inplace=act_inplace))

        layers.append(ConvSC(C_hid, C_hid, spatio_kernel,
                             upsampling=samplings[-1],
                             act_inplace=act_inplace))

        self.dec = nn.Sequential(*layers)
        self.readout = nn.Conv2d(C_hid, C_out, kernel_size=1)

    def forward(self, hid, enc1=None):
        x = hid
        for i in range(0, len(self.dec) - 1):
            x = self.dec[i](x)

        if enc1 is not None:
            x = self.dec[-1](x + enc1)
        else:
            x = self.dec[-1](x)

        y = self.readout(x)
        return y
"""