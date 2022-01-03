"""
    Custom layers for hyperspectral image models. Adapted from fastai sources
"""

from fastai.torch_core import *
from fastai.layers import SelfAttention

NormType = Enum('NormType', 'Batch BatchZero Weight Spectral')


__all__ = ['AdaptiveConcatPool3d', 'conv3d', 'conv_layer', 'conv3d_trans', 'batchnorm_3d', 'relu', 'GeneralRelu', 'Mish']

class Mish(nn.Module):
    "Ref: https://arxiv.org/ftp/arxiv/papers/1908/1908.08681.pdf"
    def __init__(self, *kwargs):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

class GeneralRelu(nn.Module):
    "Class that can be used as Relu, Leaky Relu, subtracted Relu or Relu<number>"
    def __init__(self, leak=None, sub=None, maxv=None):
        super().__init__()
        self.leak,self.sub,self.maxv = leak,sub,maxv

    def forward(self, x): 
        x = F.leaky_relu(x,self.leak) if self.leak is not None else F.relu(x)
        if self.sub is not None: x.sub_(self.sub)
        if self.maxv is not None: x.clamp_max_(self.maxv)
        return x


def relu(inplace:bool=False, leaky:float=None):
    "Return a relu activation, maybe `leaky` and `inplace`."
    return nn.LeakyReLU(inplace=inplace, negative_slope=leaky) if leaky is not None else nn.ReLU(inplace=inplace)


def conv3d(ni:int, nf:int, ks:int=3, stride:int=1, padding:int=None, bias=False, 
           init:LayerFunc=nn.init.kaiming_normal_) -> nn.Conv3d:
    "Create and initialize `nn.Conv3d` layer. `padding` defaults to `ks//2`."
    if padding is None: padding = ks//2
    return init_default(nn.Conv3d(ni, nf, kernel_size=ks, stride=stride, padding=padding, bias=bias), init)


def conv3d_trans(ni:int, nf:int, ks:int=2, stride:int=2, padding:int=0, bias=False) -> nn.ConvTranspose3d:
    "Create `nn.ConvTranspose3d` layer."
    return nn.ConvTranspose3d(ni, nf, kernel_size=ks, stride=stride, padding=padding, output_padding=(1,0,0), bias=bias)


def conv_layer(ni:int, nf:int, ks:int=3, stride:int=1, padding:int=None, bias:bool=None, dims:int=2,
               norm_type:Optional[NormType]=NormType.Batch, activ:Optional[nn.Module]=GeneralRelu,
               use_activ:bool=True, leaky:float=None, sub:float=None, maxv:float=None,
               transpose:bool=False, init:Callable=nn.init.kaiming_normal_, self_attention:bool=False):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and batchnorm (if `bn`) layers."
    if padding is None: padding = (ks-1)//2 if not transpose else 0
    bn = norm_type in (NormType.Batch, NormType.BatchZero)
    if bias is None: bias = not bn
    if transpose:
        conv_func = nn.ConvTranspose2d if dims == 2 else partial(nn.ConvTranspose3d, output_padding=(1,0,0))
    else:
        conv_func = nn.Conv1d if dims == 1 else nn.Conv2d if dims == 2 else nn.Conv3d
    conv = init_default(conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding), init)
    if   norm_type==NormType.Weight:   conv = weight_norm(conv)
    elif norm_type==NormType.Spectral: conv = spectral_norm(conv)
    layers = [conv]
    if use_activ: layers.append(activ(leaky, sub, maxv))
    if bn: layers.append((nn.BatchNorm1d if dims==1 else nn.BatchNorm2d if dims==2 else nn.BatchNorm3d)(nf)) 
    if self_attention: layers.append(SelfAttention(nf))
    return nn.Sequential(*layers)


def batchnorm_3d(nf:int, norm_type:NormType=NormType.Batch):
    "A batchnorm3d layer with `nf` features initialized depending on `norm_type`."
    bn = nn.BatchNorm3d(nf)
    with torch.no_grad():
        bn.bias.fill_(1e-3)
        bn.weight.fill_(0. if norm_type==NormType.BatchZero else 1.)
    return bn


class AdaptiveConcatPool3d(Module):
    "Layer that concats `AdaptiveAvgPool3d` and `AdaptiveMaxPool3d`."
    def __init__(self, sz:Optional[int]=None):
        "Output will be 2*sz or 2 if sz is None"
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool3d(self.output_size)
        self.mp = nn.AdaptiveMaxPool3d(self.output_size)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

class MultiScaleBlock(nn.Module):
    "TODO: Add options to set blocks during init"
    def __init__(self, in_channels, out_channels, bias, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, 
                               kernel_size=(1,1,1), bias=bias, padding=(0,0,0))
        self.conv3 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, 
                               kernel_size=(3,1,1), bias=bias, padding=(1,0,0))
        self.conv5 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, 
                               kernel_size=(5,1,1), bias=bias, padding=(2,0,0))
        self.conv11 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, 
                               kernel_size=(11,1,1), bias=bias, padding=(5,0,0))
    def forward(self, x):
        out_1 = self.conv1(x)
        out_3 = self.conv3(x)
        out_5 = self.conv5(x)
        out_11 = self.conv11(x)
        return out_1 + out_3 + out_5 + out_11
