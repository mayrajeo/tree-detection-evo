from fastai import *
from fastai.vision import *
import src.customlayers as cl
from fastai.torch_core import *
from fastai.core import *
from fastai.layers import ResizeBatch
import math

__all__ = ['simple_cnn_cust', 'calc_shape_2d', 'calc_shape_3d', 'hybrid_cnn', 'conv_1d_net']

def conv_1d_net(in_c, out_c, bn:bool=False, activ:nn.Module=cl.GeneralRelu, leaky=None, maxv=None, sub=None,
                drop:float=0.0) -> nn.Sequential:
    "Modified from https://www.tandfonline.com/doi/full/10.1080/22797254.2018.1434424. TODO add BN"
    lin_c = int(math.floor(((in_c - 17 + 1)-1*(4-1) -1 )/4) + 1)
    bn_1 = nn.BatchNorm1d(lin_c * 20) if bn else Lambda(lambda x:x)
    bn_2 = nn.BatchNorm1d(100) if bn else Lambda(lambda x:x)
    meanlayer = Lambda(lambda x: x.mean(dim=(2,3))[:,None])
    model = nn.Sequential( # Remove when dataclass handles this.
                          meanlayer,
                          nn.Conv1d(1, 20, 17),
                          activ(leak=leaky, maxv=maxv, sub=sub),
                          nn.MaxPool1d(4, None),
                          Flatten(),
                          bn_1,
                          nn.Linear(20*lin_c, 100, bias=not bn), 
                          activ(leak=leaky, maxv=maxv, sub=sub),
                          nn.Dropout(drop),
                          bn_2, 
                          nn.Linear(100, out_c, bias=not bn))
    return model

def simple_cnn_cust(actns:Collection[int], kernel_szs:Collection[int]=None,
                    strides:Collection[int]=None, bn=False, dims=2, paddings:Collection[int]=None,
                    pooling=True, activ:Optional[nn.Module]=cl.GeneralRelu, leaky=None, 
                    maxv=None, sub=None) -> nn.Sequential:
    nl = len(actns)-1
    kernel_szs = ifnone(kernel_szs, [3]*nl)
    strides = ifnone(strides, [2]*nl)
    paddings = ifnone(paddings, [None]*nl)
    layers = [cl.conv_layer(actns[i], actns[i+1], kernel_szs[i], stride=strides[i], dims=dims,
              norm_type=(cl.NormType.Batch if bn and i<(len(strides)-1) else None), activ=activ, 
              leaky=leaky, maxv=maxv, sub=sub, padding=paddings[i]) for i in range_of(strides)]
    if pooling:
        if dims == 3: 
            layers.append(cl.AdaptiveConcatPool3d(1))
        else: 
            layers.append(AdaptiveConcatPool2d(1))
        layers.append(Flatten())
    return nn.Sequential(*layers)

def calc_shape_2d(input_shape:Collection[int], kernel_szs:Collection[int], strides:Collection[int], 
                  paddings:Collection[int], dilation:int=1):
    "Calculate output shape from series of 2d cnn layers"
    h, w = input_shape
    for i in range_of(strides):
        if type(kernel_szs[i]) is not tuple:
            kernel_size = (kernel_szs[i], kernel_szs[i])
        else: kernel_size = kernel_szs[i]
        if type(strides[i]) is not tuple:
            stride = (strides[i], strides[i])
        else: stride = strides[i]
        if type(paddings[i]) is not tuple:
            pad = (paddings[i], paddings[i])
        else: pad = paddings[i]
        
        h = (h + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1)// stride[0] + 1
        w = (w + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1)// stride[1] + 1
    return h, w

def calc_shape_3d(input_shape:Collection[int], kernel_szs:Collection[int], strides:Collection[int], 
                  paddings:Collection[int], dilation:int=1):
    "Calculate output shape from series of 3d cnn layers"
    d, h, w = input_shape
    for i in range_of(strides):
        if type(kernel_szs[i]) is not tuple:
            kernel_size = (kernel_szs[i], kernel_szs[i], kernel_szs[i])
        else: kernel_size = kernel_szs[i]
        if type(strides[i]) is not tuple:
            stride = (strides[i], strides[i], strides[i])
        else: stride = strides[i]
        if type(paddings[i]) is not tuple:
            pad = (paddings[i], paddings[i], paddings[i])
        else: pad = paddings[i]
        d = (d + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1)// stride[0] + 1
        h = (h + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1)// stride[1] + 1
        w = (w + (2 * pad[2]) - (dilation * (kernel_size[2] - 1)) - 1)// stride[2] + 1
    return d, h, w

def hybrid_cnn(in_shape:Collection[int], out_c:int, actns_3d:Collection[int], actns_2d:Collection[int], 
               krnls_3d:Collection[int]=None, krnls_2d:Collection[int]=None, 
               strides_3d:Collection[int]=None, strides_2d:Collection[int]=None, 
               pads_3d:Collection[int]=[0], pads_2d:Collection[int]=[0], bn:bool=False, 
               leaky=None, maxv=None, sub=None, drop=0.5) -> nn.Sequential:
    """Create a hybrid 3d-2d cnn similar to  https://arxiv.org/pdf/1902.06701v3.pdf, 
    but with adaptiveconcatpool instead of flatten+dense"""
    model_3d = simple_cnn_cust(actns_3d, bn=bn, dims=3, strides=strides_3d, paddings=pads_3d, kernel_szs=krnls_3d,                                pooling=False, leaky=leaky, maxv=maxv, sub=sub)
    out_shape = calc_shape_3d(in_shape, krnls_3d, strides_3d, pads_3d)
    reshape = ResizeBatch(actns_3d[-1]*out_shape[0], out_shape[1], out_shape[2])
    bn_interm = nn.BatchNorm2d(actns_3d[-1]*out_shape[0]) if bn else Lambda(lambda x:x)
    model_2d = simple_cnn_cust([actns_3d[-1]*out_shape[0]] + actns_2d, bn=bn, dims=2, 
                               strides=strides_2d, paddings=pads_2d, 
                               kernel_szs=krnls_2d, pooling=False, leaky=leaky, maxv=maxv, sub=sub)
    out_2d = calc_shape_2d((out_shape[1], out_shape[2]), krnls_2d, strides_2d, pads_2d)
    out_2d = actns_2d[-1]*out_2d[0]*out_2d[1]
    bn_1d_1 = nn.BatchNorm1d(out_2d) if bn else Lambda(lambda x:x)
    bn_1d_2 = nn.BatchNorm1d(512) if bn else Lambda(lambda x:x)
    bn_1d_3 = nn.BatchNorm1d(256) if bn else Lambda(lambda x:x)
    model = nn.Sequential(model_3d, reshape, bn_interm, model_2d, 
                          #AdaptiveConcatPool2d(1), 
                          Flatten(),
                          bn_1d_1, 
                          nn.Linear(out_2d, 512, bias=not bn), cl.GeneralRelu(leaky, maxv, sub), 
                          nn.Dropout(drop), bn_1d_2,
                          nn.Linear(512, 256, bias=not bn), cl.GeneralRelu(leaky, maxv, sub), 
                          nn.Dropout(drop), bn_1d_3,
                          nn.Linear(256, out_c))
    return model

def pol_ann_etal_3d(out_c:int, drop:float=0.25, bn:bool=False, leaky:float=None, maxv:float=None, sub:float=None):
    "Reference: https://ieeexplore.ieee.org/document/8747253/"
    # TODO
    norm_type = None if not bn else NormType.Batch
    model = nn.Sequential(
        cl.conv_layer(ni=6, nf=64, ks=(1,3,3), stride=1, dims=3, norm_type=norm_type, 
                      leaky=leaky, maxv=maxv, sub=sub),
        cl.conv_layer(ni=64, nf=64, ks=(3,3,3), stride=1, dims=3, norm_type=norm_type, 
                      leaky=leaky, maxv=maxv, sub=sub),
        nn.MaxPool3d((1,2,2)),
        cl.conv_layer(ni=64, nf=128, ks=(3,3,3), stride=1, dims=3, norm_type=norm_type, 
                      leaky=leaky, maxv=maxv, sub=sub),
        nn.MaxPool3d((3,2,2)),
        cl.conv_layer(ni=128, nf=256, ks=(3,3,3), stride=1, dims=3, norm_type=norm_type, 
                      leaky=leaky, maxv=maxv, sub=sub),
        nn.MaxPool3d((3,2,2)),
        Flatten(),
        nn.Linear(256, 128, bias=False), cl.GeneralRelu(sub=sub, leak=leaky, maxv=maxv), nn.BatchNorm1d(128),
        nn.Dropout(drop), nn.Linear(128, out_c))
    return model


class SimpleNet(Module):
    "Simple baseclass for network that has (features, classifier) -structure"
    def __init__(self, features, classifier):
        #super(SimpleNet, self).__init__()
        self.features = features
        self.classifier = classifier
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

class Autoencoder(nn.Module):
    "TODO Scale all data to range 0-1"
    def __init__(self, actns, kernel_szs, strides, pads, bn=False):
        super(Autoencoder, self).__init__()
        enc_layers = [cl.conv_layer(actns[i], actns[i+1], kernel_szs[i], dims=3, padding=pads[i], 
                      stride=strides[i], norm_type=cl.NormType.Batch if bn else None 
                      ) for i in range_of(strides)]
        self.encoder = nn.Sequential(*enc_layers)
        dec_layers = [cl.conv_layer(actns[i+1], actns[i], kernel_szs[i], transpose=True, dims=3, padding=pads[i],
                      stride=strides[i], norm_type=cl.NormType.Batch if bn else None 
                      ) for i in range(len(strides)-1, 0, -1)]
        self.decoder = nn.Sequential(*dec_layers, nn.ConvTranspose3d(actns[1], actns[0], 
                                     kernel_size=kernel_szs[0], 
                                     stride=strides[0], padding=pads[0]), nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

