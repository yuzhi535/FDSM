import torch
import torch.nn as nn
import math
import numpy as np
import torch.fft
from timm.models.vision_transformer import Mlp

# DCT and IDCT implementations
def dct(x, norm='ortho'):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    Applied to the last dimension.
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.fft.fft(v, dim=1)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc.real * W_r - Vc.imag * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm='ortho'):
    """
    The Inverse Discrete Cosine Transform, Type II (a.k.a. the IDCT)
    Applied to the last dimension.
    """
    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, N) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(N, dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = torch.fft.irfft(torch.view_as_complex(V), n=N, dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


class FrequencyEnhancement(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, gate=None):
        # x: (B, N, C)
        x_freq = dct(x)
        N = x.shape[-1]
        M = N // 4  # Matches paper M=L/4
        
        # Enhance high frequencies: (1 + alpha * gate)
        # If gate is None, assume gate=1 (uniform enhancement) or 0? 
        # Paper implies gate comes from semantic intensity.
        if gate is not None:
            # gate shape expected: (B, 1) or (B)
            # We need gain shape: (B, 1, 1) to broadcast against x_freq: (B, N, C)
            if gate.dim() == 1:
                gate = gate.view(-1, 1, 1)
            elif gate.dim() == 2:
                # If (B, 1), make it (B, 1, 1)
                gate = gate.unsqueeze(-1)
            
            gain = 1.0 + self.alpha * gate
            
            # Avoid in-place modification: x_freq[:, :, M:] = ...
            low_freq = x_freq[:, :, :M]
            high_freq = x_freq[:, :, M:] * gain
            x_freq = torch.cat([low_freq, high_freq], dim=-1)
        else:
            # Fallback to static enhancement if no gate provided (or throw error?)
            # Keeping previous behavior roughly (static weight) if alpha matches
            # But here we stick to the new formula. 
            pass 

        out = idct(x_freq)
        return out


class KinematicProjectionHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C) - text embedding
        return self.net(x)


''' Multi-Head Self Attention (MHSA) '''
class MHSA(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, qk_norm: bool = False,
                 attn_drop: float = 0., proj_drop: float = 0., norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

        self.qkv_c = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm_c = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm_c = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj_c = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # Linear
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        B, N_c, C = cond.shape
        qkv_c = self.qkv_c(cond).reshape(B, N_c, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # Linear
        q_c, k_c, v_c = qkv_c.unbind(0)
        q_c, k_c = self.q_norm_c(q_c), self.k_norm_c(k_c)

        # Token-wise concatenate
        q = torch.cat((q, q_c), dim=-2)
        k = torch.cat((k, k_c), dim=-2)
        v = torch.cat((v, v_c), dim=-2)

        # Multi-head self-attention
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v

        # Token-specific split
        out = out.transpose(1, 2).reshape(B, N + N_c, C)
        x, cond = out[:, :N], out[:, N:]

        x = self.proj(x)
        x = self.proj_drop(x)

        cond = self.proj_c(cond)
        cond = self.proj_drop(cond)
        return x, cond


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)  # Scale-Shift


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size, bias=True), nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


''' CrossDiT Block '''
class CrossDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_freq_enhance=True, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.norm1_c = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2_c = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = MHSA(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.mlp_c = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.adaLN_modulation_c = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.use_freq_enhance = use_freq_enhance
        if self.use_freq_enhance:
            self.freq_enhance = FrequencyEnhancement(alpha=1.0)

    def forward(self, x, cond, c, gate=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        shift_msa_c, scale_msa_c, gate_msa_c, shift_mlp_c, scale_mlp_c, gate_mlp_c = self.adaLN_modulation_c(c).chunk(6, dim=1)

        x_temp = modulate(self.norm1(x), shift_msa, scale_msa)  # Scale-Shift
        cond_temp = modulate(self.norm1_c(cond), shift_msa_c, scale_msa_c)  # Scale-Shift

        x_temp, cond_temp = self.attn(x_temp, cond_temp)  # MHSA

        x = x + gate_msa.unsqueeze(1) * x_temp  # Scale
        if self.use_freq_enhance:
            x = self.freq_enhance(x, gate=gate)
        cond = cond + gate_msa_c.unsqueeze(1) * cond_temp  # Scale

        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))  # Scale-Shift-FeedForward-Scale
        cond = cond + gate_mlp_c.unsqueeze(1) * self.mlp_c(modulate(self.norm2_c(cond), shift_mlp_c, scale_mlp_c))  # Scale-Shift-FeedForward-Scale
        return x, cond


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)  # Scale-Shift
        x = self.linear(x)
        return x


''' Diffusion Transformer '''
class DiT(nn.Module):
    def __init__(self, in_channels=256, cond_size=2048, hidden_size=768, depth=12, num_heads=12, mlp_ratio=4.0, use_freq_enhance=True):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads

        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.fc_embedder = nn.Sequential(nn.Linear(cond_size, hidden_size, bias=True), nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True))
        self.fl_embedder = nn.Linear(cond_size, hidden_size, bias=True)
        self.fl_pos_embed = nn.Parameter(torch.zeros(1, 35, hidden_size), requires_grad=True)  # M_l = 35

        self.blocks = nn.ModuleList([CrossDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_freq_enhance=use_freq_enhance) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, in_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize positional embedding table:
        nn.init.normal_(self.fl_pos_embed, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out AdaLN modulation layers in CrossDiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.adaLN_modulation_c[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation_c[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, fc, fl, gate=None):
        x = self.x_embedder(x)  # (B, M_x , D)
        fl = self.fl_embedder(fl) + self.fl_pos_embed  # (B, M_l, D)
        t = self.t_embedder(t)  # (B, D)
        fc = self.fc_embedder(fc)  # (B, D)
        fc = t + fc
        for block in self.blocks:
            x, fl = block(x, fl, fc, gate=gate)  # (B, M_x, D)
        x = self.final_layer(x, fc)
        return x