import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .vit_nlc import Encoder, Decoder


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std *torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])

                # return 0.5 * torch.mean(torch.pow(self.mean, 2)
                #                         + self.var - 1.0 - self.logvar,
                #                         dim=[1, 2, 3])
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean



class AutoEncoder(nn.Module):
    """
    Args:
    N (int): Number of channels
    M (int): Number of channels in the expansion layers (last layer of the
    encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, 
                 model_version,
                 embed_dim=None, 
                 y_channels=None,
                 sample_posterior=None, 
                 ddconfig=None, 
                 lower_dim= False,
                 **kwargs):
        if model_version == 268:
            embed_dim=256
            y_channels=1024
            lower_dim=True
            sample_posterior =True
            ddconfig=dict(
                arch = 'vit_large',
                pretrained_model = '',
                patch_size=(11,10),
                patch_stride=(10,10),
                in_chans=268,
                out_chans=268,
                kwargs=dict(
                    z_dim =  None,
                    learnable_pos= True,
                    window= True,
                    window_size = [(24, 24), (12, 48), (48, 12)],
                    interval = 4,
                    drop_path_rate= 0.,
                    round_padding= True,
                    pad_attn_mask= True ,    
                    test_pos_mode= 'learnable_simple_interpolate',    
                    img_size= (721, 1440)
                ),
            )

        if model_version == "34_4":
            embed_dim=34
            y_channels=1024
            lower_dim=True
            sample_posterior = False
            ddconfig=dict(
                arch = 'vit_large',
                pretrained_model = '',
                patch_size=(4,4),
                patch_stride=(4,4),
                in_chans=69,
                out_chans=69,
                kwargs=dict(
                    z_dim =  None,
                    learnable_pos= True,
                    window= True,
                    window_size = [(24, 24), (12, 48), (48, 12)],
                    interval = 4,
                    drop_path_rate= 0.,
                    round_padding= True,
                    pad_attn_mask= True ,    
                    test_pos_mode= 'learnable_simple_interpolate',    
                    img_size= (128, 256)
                ),
        )


        super().__init__(**kwargs)
        self.sample_posterior = sample_posterior
        self.lower_dim = lower_dim
        
        self.g_a = Encoder(**ddconfig)
        self.g_s = Decoder(**ddconfig)
        
        self.quant_conv = torch.nn.Conv2d(2*y_channels, 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, y_channels, 1)

    def encode(self, x):
        moments = self.g_a(x)
        posterior = None

        moments = self.quant_conv(moments)

        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        x_hat = self.g_s(z)

        return x_hat

    def forward(self, x):

        posterior = self.encode(x)
        if self.sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        x_hat = self.decode(z)

        return x_hat, posterior
