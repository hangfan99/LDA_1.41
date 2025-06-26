from turtle import forward
import torch
import torch.nn as nn
from networks.utils.utils import DropPath, PeriodicPad2d, Mlp
from networks.utils.Attention import SD_attn, WindowAttention, HiLo, Conv_attn
from networks.utils.mlp import DWMlp
from networks.utils.Attention import SD_attn_withmoe
from networks.utils.mlp import Mlp_withmoe



class Convnet_block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, kernel_size=[4, 8], drop_path=0., layer_scale_init_value=1e-6, norm_layer=nn.LayerNorm):
        super().__init__()
        padding_size = [i // 2 for i in kernel_size]
        self.padding = PeriodicPad2d(padding_size)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=0, groups=12) # depthwise conv
        self.norm = norm_layer(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = x.permute(0, 3, 1, 2)
        x = self.padding(x)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x

        x = input + self.drop_path(x)
        return x


class Originattn_block(nn.Module):
    def __init__(self, dim, window_size, num_heads=1, mlp_ratio=4., 
                qkv_bias=True, drop=0., attn_drop=0., drop_path=0., 
                act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                attn_type="windowattn", pre_norm=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.pre_norm = pre_norm

        self.norm = norm_layer(dim)
        # self.GAU1 = Flash_attn(dim, window_size=self.window_size, uv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, expansion_factor=2, attn_type='lin')

        if attn_type == "origin_attn":
            self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=attn_drop, bias=qkv_bias, batch_first=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    

    

    def forward(self, x):
        shortcut = x
        # partition windows

        if self.pre_norm:
            x = self.norm(x)
            x = shortcut + self.drop_path(self.attn(x, x, x)[0])
        else:
            x = self.norm(shortcut + self.drop_path(self.attn(x, x, x)[0]))

        # W-MSA/SW-MS

        if self.pre_norm:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = self.norm2(x + self.drop_path(self.mlp(x)))


        return x





class Windowattn_block(nn.Module):
    def __init__(self, dim, window_size, num_heads=1, mlp_ratio=4., 
                qkv_bias=True, drop=0., attn_drop=0., drop_path=0., 
                act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                attn_type="windowattn", pre_norm=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.pre_norm = pre_norm

        self.norm = norm_layer(dim)
        # self.GAU1 = Flash_attn(dim, window_size=self.window_size, uv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, expansion_factor=2, attn_type='lin')
        if attn_type == "windowattn":
            if "shift_size" not in kwargs:
                shift_size = [0, 0, 0]
            else:
                shift_size = kwargs["shift_size"]
            if "dilated_size" in kwargs:
                dilated_size = kwargs["dilated_size"]
            else:
                dilated_size = [1, 1, 1]
            self.attn = SD_attn(
                dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias,
                attn_drop=attn_drop, proj_drop=drop, shift_size=shift_size, dilated_size=dilated_size)
        elif attn_type == "convattn":
            self.attn = Conv_attn(dim, window_size, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        elif attn_type == "origin_attn":
            self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=attn_drop, bias=qkv_bias, batch_first=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    

    

    def forward(self, x):
        shortcut = x
        # partition windows

        if self.pre_norm:
            x = shortcut + self.drop_path(self.attn(self.norm(x)))
        else:
            x = self.norm(shortcut + self.drop_path(self.attn(x)))

        # W-MSA/SW-MS

        if self.pre_norm:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = self.norm2(x + self.drop_path(self.mlp(x)))


        return x


class Hilo_Block(nn.Module):
    def __init__(self, dim, window_size, num_heads=1, mlp_ratio=4., 
                qkv_bias=True, drop=0., attn_drop=0., drop_path=0., 
                act_layer=nn.ReLU, norm_layer=nn.LayerNorm, pre_norm=True,
                alpha=0.9) -> None:
        super().__init__()
        self.dim = dim
        self.window_size=window_size
        self.pre_norm = pre_norm

        self.norm1 = norm_layer(dim)
        self.attn = HiLo(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                        proj_drop=drop, window_size=window_size, alpha=alpha)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = DWMlp(dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
    def forward(self, x):
        shortcut = x
        # partition windows

        if self.pre_norm:
            x = shortcut + self.drop_path(self.attn(self.norm1(x)))
        else:
            x = self.norm1(shortcut + self.drop_path(self.attn(x)))

        # W-MSA/SW-MS

        if self.pre_norm:
            x = x + self.drop_path(self.convffn(self.norm2(x)))
        else:
            x = self.norm2(x + self.drop_path(self.convffn(x)))
        return x

class ConvFFNBlock(nn.Module):
    """ Convolutional FFN Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, window_size=7, num_heads=1, 
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, alpha=0.5):
        super().__init__()
        self.dim = dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DWMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Windowattn_block_withmoe(nn.Module):
    def __init__(self, dim, attr_len, window_size, attr_hidden_size, num_heads=1, mlp_ratio=4., 
                qkv_bias=True, drop=0., attn_drop=0., drop_path=0., 
                act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                attn_type="windowattn", pre_norm=True, attn_use_moe=True, 
                mlp_use_moe=True, num_experts=1, expert_capacity=1., router_bias=True, 
                router_noise=1e-2, is_scale_prob=True, drop_tokens=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.pre_norm = pre_norm
        self.attn_use_moe = attn_use_moe
        self.mlp_use_moe = mlp_use_moe

        self.norm = norm_layer(dim)
        # self.GAU1 = Flash_attn(dim, window_size=self.window_size, uv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, expansion_factor=2, attn_type='lin')
        if attn_type == "windowattn":
            if "shift_size" not in kwargs:
                shift_size = [0, 0, 0]
            else:
                shift_size = kwargs["shift_size"]
            if "dilated_size" in kwargs:
                dilated_size = kwargs["dilated_size"]
            else:
                dilated_size = [1, 1, 1]
            if attn_use_moe:
                self.attn = SD_attn_withmoe(
                    dim, attr_len=attr_len, attr_hidden_size=attr_hidden_size, window_size=self.window_size, num_heads=num_heads,
                    qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, shift_size=shift_size, dilated_size=dilated_size,
                    num_experts=num_experts, expert_capacity=expert_capacity, router_bias=router_bias, router_noise=router_noise,
                    is_scale_prob=is_scale_prob, drop_tokens=drop_tokens
                )
            else:
                self.attn = SD_attn(
                    dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias,
                    attn_drop=attn_drop, proj_drop=drop, shift_size=shift_size, dilated_size=dilated_size)
        elif attn_type == "convattn":
            raise NotImplementedError('moe convattn')

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if mlp_use_moe:
            self.mlp = Mlp_withmoe(in_features=dim, attr_len=attr_len, attr_hidden_size=attr_hidden_size, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop, num_experts=num_experts, expert_capacity=expert_capacity, router_bias=router_bias,
            router_noise=router_noise, is_scale_prob=is_scale_prob, drop_tokens=drop_tokens)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    

    def forward(self, x, attr=None):
        shortcut = x
        # partition windows

        if self.pre_norm:
            x = self.norm(x)
            if self.attn_use_moe:
                x, z_loss1, balance_loss1 = self.attn(x, attr)
            else:
                x = self.attn(x)
                z_loss1, balance_loss1 = 0, 0
            x = shortcut + self.drop_path(x)
            shortcut = x
            x = self.norm2(x)
            if self.mlp_use_moe:
                x, z_loss2, balance_loss2 = self.mlp(x, attr)
            else:
                x = self.mlp(x)
                z_loss2, balance_loss2 = 0, 0
            x = shortcut + self.drop_path(x)
        else:
            if self.attn_use_moe:
                x, z_loss1, balance_loss1 = self.attn(x, attr)
            else:
                x = self.attn(x)
                z_loss1, balance_loss1 = 0, 0
            x = self.norm(shortcut + self.drop_path(x))
            shortcut = x
            if self.mlp_use_moe:
                x, z_loss2, balance_loss2 = self.mlp(x, attr)
            else:
                x = self.mlp(x)
                z_loss2, balance_loss2 = 0, 0
            x = self.norm2(shortcut + self.drop_path(x))

        return x, [z_loss1, z_loss2], [balance_loss1, balance_loss2]




