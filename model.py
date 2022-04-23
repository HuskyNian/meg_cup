import megengine.functional as torch
import megengine as meg
import megengine.functional.nn as F
import megengine.module as nn

import numpy as np
#import torch.nn as nn
#import torch
#import torch.nn.functional as F
#from torch.utils.data import Dataset
from collections import OrderedDict
import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm
import os.path as osp
import gc


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)




class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=True, act=nn.ReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU())
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU())
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.shape[2], x.shape[3]), stride=(x.shape[2], x.shape[3]))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.shape[2], x.shape[3]), stride=(x.shape[2], x.shape[3]))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.shape[2], x.shape[3]), stride=(x.shape[2], x.shape[3]))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = F.broadcast_to(F.expand_dims(F.sigmoid(channel_att_sum), (2, 3)), x.shape)
        #scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.shape[0], tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, axis=2, keepdims=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.concat((torch.expand_dims(torch.max(x, 1), 1), torch.expand_dims(torch.mean(x, 1), 1)), axis=1)
        #return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

import numbers



##########################################################################
## Layer Norm

def to_3d(x):
    b,c,h,w = x.shape
    x = torch.transpose(x,(0,2,3,1))
    x = x.reshape(b,h*w,c)
    #return rearrange(x, 'b c h w -> b (h w) c')
    return x

def to_4d(x,h,w):
    b,_,c = x.shape
    x = torch.transpose(x,(0,2,1))  # b c (h w)
    x = x.reshape(b,c,h,w)
    #return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)
    return x

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.hidden_features = hidden_features
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        #x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x12 = self.dwconv(x)
        #print('x12',x12.shape)
        x1,x2 = x12[:,:self.hidden_features],x12[:,self.hidden_features:self.hidden_features*2]
        #print(x1.shape,x2.shape)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        #self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature = meg.Parameter(torch.ones([num_heads, 1, 1]))
        self.dim = dim
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv[:,:self.dim,:,:],qkv[:,self.dim:self.dim*2,:,:],qkv[:,self.dim*2:self.dim*3,:,:]
        #q,k,v = qkv.chunk(3,dim=1)
        q = q.reshape(b,c,h*w).reshape(b,self.num_heads,c//self.num_heads,h*w)
        k = k.reshape(b,c,h*w).reshape(b,self.num_heads,c//self.num_heads,h*w)
        v = v.reshape(b,c,h*w).reshape(b,self.num_heads,c//self.num_heads,h*w)
        #q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        #k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        #v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        #q = torch.nn.functional.normalize(q, dim=-1)
        #k = torch.nn.functional.normalize(k, dim=-1)
        
        q = torch.normalize(q, axis=-1)
        k = torch.normalize(k, axis=-1)
        
        
        #attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = (q @ torch.transpose(k,(0,1,3,2))) * self.temperature
        #attn = attn.softmax(dim=-1)
        attn = F.softmax(attn,axis=-1)

        out = (attn @ v)
        
        #out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.reshape(b,c,h,w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 =nn.Identity()#LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 =nn.Identity()#LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- Restormer -----------------------

def rand_bbox(size, cut_rat):
    W = size[2]
    H = size[3]
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
def cutmix(input,target):
    rand_index = torch.randperm(input.size()[0])
    target_a = target
    target_b = target[rand_index]
    cut_ratio = np.random.uniform()
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), cut_ratio)
    input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
    target[:, :, bbx1:bbx2, bby1:bby2] = target[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
   
    return input,target
class TCond(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.t_embedding = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.x_embedding = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x, t):
        t = self.t_embedding(t)
        scale = self.x_embedding(x.mean((-1, -2)) + t) * 0.1
        return scale
class Restormer_lite(nn.Module):
    def __init__(self, 
        inp_channels=4, 
        out_channels=4, 
        dim = 48,
        num_blocks = [1,1,1], 
        num_refinement_blocks = 1,
        heads = [2,8,8],
        ffn_expansion_factor = 2.05,
        num_recur = 4,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(Restormer_lite, self).__init__()
        self.num_recur = num_recur
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        #self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        #self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        #self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        #self.reduce_chan_level1 = nn.Conv2d(int(dim*2**1), int(dim), kernel_size=1, bias=bias)
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        self.attn_list = [
            CBAM(
                dim, reduction_ratio=2
            ) for _ in range(self.num_recur)
        ]
            
        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.t_cond = TCond(dim)

    def forward(self, x):
        n, c, h, w = x.shape
        #x = x.reshape((n, c, h // 2, 2, w // 2, 2)).permute((0, 1, 3, 5, 2, 4))
        x = torch.transpose(x.reshape((n, c, h // 2, 2, w // 2, 2))   ,(0, 1, 3, 5, 2, 4))
        inp_img = x.reshape((n, c * 4, h // 2, w // 2))
        inp_enc_level1 = self.patch_embed(inp_img)
        
        for i in range(self.num_recur):  
            inp_enc_level1 = self.attn_list[i](inp_enc_level1)   
            out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
            #inp_enc_level2 = self.down1_2(out_enc_level1)
            #out_enc_level2 = self.encoder_level2(inp_enc_level2)

            #inp_dec_level1 = self.up2_1(out_enc_level2)
            #inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
            #inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
            out_dec_level1 = self.decoder_level1(out_enc_level1)
            out_dec_level1 = self.refinement(out_dec_level1)
            t_cond_scale = self.t_cond(inp_enc_level1,torch.full((len(x),1),(i+1)/self.num_recur,device=inp_enc_level1.device))
            #inp_enc_level1 = inp_enc_level1 + out_dec_level1 * (t_cond_scale.unsqueeze(2).unsqueeze(3)+1)
            inp_enc_level1 = inp_enc_level1 + out_dec_level1 *( F.expand_dims(t_cond_scale,(2,3))  +1)
            
        
        
        out_dec_level1 = self.output(inp_enc_level1) + inp_img

        #out_dec_level1 = out_dec_level1.reshape((n, c, 2, 2, h // 2, w // 2)).permute((0, 1, 4, 2, 5, 3))
        out_dec_level1 = torch.transpose(out_dec_level1.reshape((n, c, 2, 2, h // 2, w // 2)), (0, 1, 4, 2, 5, 3))
        out_dec_level1 = out_dec_level1.reshape((n, c, h, w))

        return out_dec_level1
    
if __name__ == '__main__':
    net = Restormer_lite()
    #meg.save(net.state_dict(), "dbg_cbam.pth")

    x = torch.zeros((2, 1, 256, 256))
    out = net(x)
    print('out',out.shape)
