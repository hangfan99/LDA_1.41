import numpy as np
import torch
import torch.nn as nn


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc


class PositionalEncodingPermute1D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x) instead of (batchsize, x, ch)
        """
        super(PositionalEncodingPermute1D, self).__init__()
        self.penc = PositionalEncoding1D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 2, 1)

    @property
    def org_channels(self):
        return self.penc.org_channels


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

    @property
    def org_channels(self):
        return self.penc.org_channels


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        # if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
        #     return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        self.cached_penc = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
        self.cached_penc = self.cached_penc.reshape(*(self.cached_penc.shape[:-1]), -1, 2)
        sin_data = torch.cat((-1*self.cached_penc[:,:,:,:,:,0].unsqueeze(-1), self.cached_penc[:,:,:,:,:,0].unsqueeze(-1)), dim=-1)
        cos_data = self.cached_penc[:,:,:,:,:,1].unsqueeze(-1)

        origin_shape = tensor.shape

        tensor = tensor.reshape(*(tensor.shape[:-1]), -1, 2)
        tensor_flip = torch.flip(tensor, dims=[-1])
        res = tensor * cos_data + tensor_flip * sin_data
        return res.reshape(*origin_shape)


class PositionalEncodingPermute3D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y, z) instead of (batchsize, x, y, z, ch)
        """
        super(PositionalEncodingPermute3D, self).__init__()
        self.penc = PositionalEncoding3D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 4, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 4, 1, 2, 3)

    @property
    def org_channels(self):
        return self.penc.org_channels


class Summer(nn.Module):
    def __init__(self, penc):
        """
        :param model: The type of positional encoding to run the summer on.
        """
        super(Summer, self).__init__()
        self.penc = penc

    def forward(self, tensor):
        """
        :param tensor: A 3, 4 or 5d tensor that matches the model output size
        :return: Positional Encoding Matrix summed to the original tensor
        """
        penc = self.penc(tensor)
        assert (
            tensor.size() == penc.size()
        ), "The original tensor size {} and the positional encoding tensor size {} must match!".format(
            tensor.size, penc.size
        )
        tensor = tensor.reshape(*(tensor.shape[:-1]), 2)
        tensor_flip = torch.flip(tensor, dims=[-1])
        tensor_flip = tensor_flip * torch.Tensor([-1, 1], device=tensor.device)
        
        return tensor + penc

#2d旋转式位置编码

class rope2(nn.Module):
    def __init__(self, shape, dim) -> None:
        super().__init__()
        
        coords_0 = torch.arange(shape[0])
        coords_1 = torch.arange(shape[1])
        coords = torch.stack(torch.meshgrid([coords_0, coords_1], indexing="ij")).reshape(2, -1)

        half_size = dim // 2
        self.dim1_size = half_size // 2
        self.dim2_size = half_size - half_size // 2
        freq_seq1 = torch.arange(0, self.dim1_size) / self.dim1_size
        freq_seq2 = torch.arange(0, self.dim2_size) / self.dim2_size
        inv_freq1 = 10000 ** -freq_seq1
        inv_freq2 = 10000 ** -freq_seq2

        sinusoid1 = coords[0].unsqueeze(-1) * inv_freq1    
        sinusoid2 = coords[1].unsqueeze(-1) * inv_freq2     

        self.sin1 = torch.sin(sinusoid1).reshape(*shape, sinusoid1.shape[-1])
        self.cos1 = torch.cos(sinusoid1).reshape(*shape, sinusoid1.shape[-1])
        self.sin2 = torch.sin(sinusoid2).reshape(*shape, sinusoid2.shape[-1])
        self.cos2 = torch.cos(sinusoid2).reshape(*shape, sinusoid2.shape[-1])


    def forward(self, x):

        self.sin1 = self.sin1.to(x.device)
        self.cos1 = self.cos1.to(x.device)
        self.sin2 = self.sin2.to(x.device)
        self.cos2 = self.cos2.to(x.device)

        x11, x21, x12, x22 = x.split([self.dim1_size, self.dim2_size, \
                                        self.dim1_size, self.dim2_size], dim=-1)
        
        res = torch.cat([x11 * self.cos1 - x12 * self.sin1, x21 * self.cos2 - x22 * self.sin2, \
                        x12 * self.cos1 + x11 * self.sin1, x22 * self.cos2 + x21 * self.sin2], dim=-1)

        return res

#3D旋转式位置编码

class rope3(nn.Module):
    def __init__(self, shape, dim) -> None:
        super().__init__()
        
        coords_0 = torch.arange(shape[0])
        coords_1 = torch.arange(shape[1])
        coords_2 = torch.arange(shape[2])
        coords = torch.stack(torch.meshgrid([coords_0, coords_1, coords_2], indexing="ij")).reshape(3, -1)

        half_size = dim // 2
        self.dim1_2_size = half_size // 3
        self.dim3_size = half_size - half_size // 3 * 2
        freq_seq1_2 = torch.arange(0, self.dim1_2_size) / self.dim1_2_size
        freq_seq3 = torch.arange(0, self.dim3_size) / self.dim3_size
        inv_freq1_2 = 10000 ** -freq_seq1_2
        inv_freq3 = 10000 ** -freq_seq3

        sinusoid1 = coords[0].unsqueeze(-1) * inv_freq1_2    
        sinusoid2 = coords[1].unsqueeze(-1) * inv_freq1_2    
        sinusoid3 = coords[2].unsqueeze(-1) * inv_freq3    

        self.sin1 = torch.sin(sinusoid1).reshape(*shape, sinusoid1.shape[-1])
        self.cos1 = torch.cos(sinusoid1).reshape(*shape, sinusoid1.shape[-1])
        self.sin2 = torch.sin(sinusoid2).reshape(*shape, sinusoid2.shape[-1])
        self.cos2 = torch.cos(sinusoid2).reshape(*shape, sinusoid2.shape[-1])
        self.sin3 = torch.sin(sinusoid3).reshape(*shape, sinusoid3.shape[-1])
        self.cos3 = torch.cos(sinusoid3).reshape(*shape, sinusoid3.shape[-1])


    def forward(self, x):

        self.sin1 = self.sin1.to(x.device)
        self.cos1 = self.cos1.to(x.device)
        self.sin2 = self.sin2.to(x.device)
        self.cos2 = self.cos2.to(x.device)
        self.sin3 = self.sin3.to(x.device)
        self.cos3 = self.cos3.to(x.device)

        x11, x21, x31, x12, x22, x32 = x.split([self.dim1_2_size, self.dim1_2_size, self.dim3_size, \
                                            self.dim1_2_size, self.dim1_2_size, self.dim3_size], dim=-1)
        
        res = torch.cat([x11 * self.cos1 - x12 * self.sin1, x21 * self.cos2 - x22 * self.sin2, x31 * self.cos3 - x32 * self.sin3, \
                        x12 * self.cos1 + x11 * self.sin1, x22 * self.cos2 + x21 * self.sin2, x32 * self.cos3 + x31 * self.sin3], dim=-1)

        return res

#相对位置编码

class RelativePositionalBias(nn.Module):
    def __init__(self, window_size, num_heads=1) -> None:
        super().__init__()

        self.total_window_size = 1
        table_len = 1
        for i in window_size:
            table_len *= 2 * i - 1
            self.total_window_size *= i

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(table_len, num_heads))  # [2*Mt-1 * 2*Mh-1 * 2*Mw-1, nH]
            
        
        # get pair-wise relative position index for each token inside the window
        coords = []
        for i in window_size:
            coords.append(torch.arange(i))

        coords = torch.stack(torch.meshgrid(coords, indexing="ij"))  # [3, Mt, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw*Mt]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw*Mt, Mh*Mw*Mt]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw*Mt, Mh*Mw*Mt, 2]

        for i in range(len(window_size)):
            relative_coords[:, :, i] += window_size[i] - 1
        for i in range(len(window_size) - 1):
            table_len = table_len // (2 * window_size[i] - 1)
            relative_coords[:, :, i] *= table_len

        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw*Mt, Mh*Mw*Mt]
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.total_window_size, self.total_window_size, -1)
            
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        x = x + relative_position_bias

        return x
