import logging
import numpy as np
import pdb
import torch
import torch.nn as nn

from modeling.torch_complex_utils import ComplexConv2d, ComplexConv3d, CReLU

KERNEL_SIZE = 3
PADDING = 'same' # [valid, same]
PADDING_MODE = 'reflect' # [zeros, reflect] Default: 'zeros'

def get_model(model_type, dimensions, n_hidden_layers=None, residual_layer=None, load_model_path=None):
    if model_type == 'dncnn':
        model = DnCNN(dimensions, 64, n_hidden_layers, residual_layer)
    elif model_type == 'cdncnn':
        model = CDnCNN(dimensions, 128, n_hidden_layers, residual_layer)
    elif model_type == 'convnext':
        model = ConvNext(dimensions, 64, n_hidden_layers, residual_layer)
    elif model_type == 'fsrcnn':
        model = FSRCNN(dimensions, 56, 12, 4)
    elif model_type == 'dcsrn':
        model = DCSRN(dimensions, fn='relu')
    elif model_type == 'my_dcsrn':
        model = myDCSRN(dimensions, fn='relu', residual_layer=residual_layer)

    if load_model_path is not None:
        logging.info(f'Loading {model_type}: {load_model_path}')
        model.load_state_dict(torch.load(load_model_path, map_location='cpu'))

    return model

def get_loss(loss):
    if loss in ['mae', 'l1']:
        loss_fn = torch.nn.L1Loss()
    elif loss in ['mse', 'l2'] and 'magnitude' in config_name:
        loss_fn = torch.nn.MSELoss()
    elif loss in ['mse', 'l2'] and 'complex' in config_name:
        def complex_mse_loss(output, target):
            '''
            Compute MSE as a complex number.
            Return magnitude of the complex number.
            '''
            tmp = ( (output - target)**2 ).mean()
            return torch.sqrt(torch.real(tmp)**2 + torch.imag(tmp)**2)
        loss_fn = complex_mse_loss
    else:
        raise NotImplementedError()

    return loss

class DnCNN(nn.Module):
    def conv_layer(self, in_features, out_features):
        if self.dimensions == 2:
            return nn.Conv2d(
                    in_features, 
                    out_features, 
                    (KERNEL_SIZE, KERNEL_SIZE),
                    stride=1,
                    padding=PADDING,
                    padding_mode=PADDING_MODE
            )
        elif self.dimensions == 3:
            return nn.Conv3d(
                    in_features, 
                    out_features, 
                    (KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE),
                    stride=1,
                    padding=PADDING,
                    padding_mode=PADDING_MODE
            )

    def batch_layer(self, out_features):
        if self.dimensions == 2:
            return nn.BatchNorm2d(out_features)
        elif self.dimensions == 3:
            return nn.BatchNorm3d(out_features)

    def __init__(self, dimensions, n_features, n_hidden_layers, residual_layer):
        super().__init__()
        self.dimensions = dimensions
        self.residual_layer = residual_layer

        self.in_conv = self.conv_layer(1, n_features)
        self.in_act  = nn.ReLU()
        
        self.hidden_layers = nn.ModuleList()
        for i in range(n_hidden_layers):
            self.hidden_layers.append(self.conv_layer(n_features, n_features))
            self.hidden_layers.append(self.batch_layer(n_features)) #tf: eps: 0.001, momentum: 0.99
            self.hidden_layers.append(nn.ReLU())

        self.out_conv = self.conv_layer(n_features, 1)
    
    def forward(self, tensor):
        x = self.in_conv(tensor)
        x = self.in_act(x)

        for h_layer in self.hidden_layers:
            x = h_layer(x)

        x = self.out_conv(x)

        if self.residual_layer: x = x + tensor

        return x

class CDnCNN(nn.Module):
    def conv_layer(self, in_features, out_features):
        if self.dimensions == 2:
            return ComplexConv2d(
                    in_features,
                    out_features,
                    KERNEL_SIZE,
                    PADDING,
                    PADDING_MODE
            )
        elif self.dimensions == 3:
            return ComplexConv3d(
                    in_features, 
                    out_features, 
                    KERNEL_SIZE, 
                    PADDING,
                    PADDING_MODE
            )

    def __init__(self, dimensions, n_features, n_hidden_layers, residual_layer):
        super().__init__()
        self.dimensions = dimensions
        self.residual_layer = residual_layer

        self.in_conv = self.conv_layer(1, n_features)
        self.in_act  = CReLU()

        self.hidden_layers = nn.ModuleList()
        for i in range(n_hidden_layers):
            self.hidden_layers.append(self.conv_layer(n_features, n_features))
            self.hidden_layers.append(CReLU())

        self.out_conv = self.conv_layer(n_features, 1)
    
    def forward(self, tensor):
        if tensor.shape[-1] == 2: tensor = torch.view_as_complex(tensor)
        x = self.in_conv(tensor)
        x = self.in_act(x)

        for h_layer in self.hidden_layers:
            x = h_layer(x)

        x = self.out_conv(x)

        if self.residual_layer: x = x + tensor

        return x

class FSRCNN(nn.Module):
    def __init__(self, dimensions, first_fm=56, mid_fm=12, mid_layers=4):
        super().__init__()
        self.dimensions = dimensions

        self.first = nn.Sequential(
                nn.Conv2d(1, first_fm, kernel_size=5, padding=PADDING, padding_mode=PADDING_MODE),
                nn.PReLU(first_fm)
        )

        self.mid = [ 
                nn.Conv2d(first_fm, mid_fm, kernel_size=1, padding=PADDING, padding_mode=PADDING_MODE),
                nn.PReLU(mid_fm)
        ]
        for _ in range(mid_layers):
            self.mid.extend([
                nn.Conv2d(mid_fm, mid_fm, kernel_size=3, padding=PADDING, padding_mode=PADDING_MODE),
                nn.PReLU(mid_fm)
            ])
        self.mid.extend([
            nn.Conv2d(mid_fm, first_fm, kernel_size=1, padding=PADDING, padding_mode=PADDING_MODE),
            nn.PReLU(first_fm)
        ])
        self.mid = nn.Sequential(*self.mid)
        self.last = nn.ConvTranspose2d(first_fm, 1, kernel_size=9, padding=PADDING, padding_mode='zeros')

    def forward(self, tensor):
        x = self.first(tensor)
        x = self.mid(tensor)
        x = self.last(tensor)

        return x

class myDCSRN(nn.Module):
    def conv_layer(self, in_features, out_features):
        if self.dimensions == 2:
            return nn.Conv2d(
                in_features,
                out_features,
                KERNEL_SIZE,
                stride=1,
                padding=PADDING,
                padding_mode=PADDING_MODE
            )
        elif self.dimensions == 3:
            return nn.Conv3d(
                in_features,
                out_features,
                KERNEL_SIZE,
                stride=1,
                padding=PADDING,
                padding_mode=PADDING_MODE
            )

    def norm_layer(self, out_features):
        if self.dimensions == 2:
            return nn.BatchNorm2d(out_features)
        elif self.dimensions == 3:
            return nn.BatchNorm3d(out_features)

    def act_fn(self, fn):
        if fn == 'relu':
            return nn.ReLU()
        elif fn == 'elu':
            return nn.ELU()

    def __init__(self, dimensions, residual_layer=True, fn='relu'):
        super().__init__()
        self.dimensions = dimensions
        self.residual_layer = residual_layer

        self.c1 = self.conv_layer(1, 2000)

        self.n1 = self.norm_layer(2000)
        self.a1 = self.act_fn(fn)
        self.c2 = self.conv_layer(2000, 48)

        self.n2 = self.norm_layer(48)
        self.a2 = self.act_fn(fn)
        self.c3 = self.conv_layer(48, 72)

        self.n3 = self.norm_layer(72)
        self.a3 = self.act_fn(fn)
        self.c4 = self.conv_layer(72, 96)

        self.n4 = self.norm_layer(96)
        self.a4 = self.act_fn(fn)
        self.c5 = self.conv_layer(96, 120)

        self.c6 = self.conv_layer(120, 1)

    def forward(self, tensor):
        CAT_AXIS = 1
        c1 = self.c1(tensor)

        x = self.n1(c1)
        x = self.a1(x)
        x = self.c2(x)

        x = self.n2(x)
        x = self.a2(x)
        x = self.c3(x)

        x = self.n3(x)
        x = self.a3(x)
        x = self.c4(x)

        x = self.n4(x)
        x = self.a4(x)
        x = self.c5(x)

        c6 = self.c6(x)

        if self.residual_layer: c6 = c6 + tensor

        return c6

class DCSRN(nn.Module):
    def conv_layer(self, in_features, out_features):
        if self.dimensions == 2:
            return nn.Conv2d(
                in_features,
                out_features,
                KERNEL_SIZE,
                stride=1,
                padding=PADDING,
                padding_mode=PADDING_MODE
            )
        elif self.dimensions == 3:
            return nn.Conv3d(
                in_features,
                out_features,
                KERNEL_SIZE,
                stride=1,
                padding=PADDING,
                padding_mode=PADDING_MODE
            )

    def norm_layer(self, out_features):
        if self.dimensions == 2:
            return nn.BatchNorm2d(out_features)
        elif self.dimensions == 3:
            return nn.BatchNorm3d(out_features)

    def act_fn(self, fn):
        if fn == 'relu':
            return nn.ReLU()
        elif fn == 'elu':
            return nn.ELU()

    def __init__(self, dimensions, fn='relu'):
        super().__init__()
        self.dimensions = dimensions

        self.c1 = self.conv_layer(1, 2000)
        self.n1 = self.norm_layer(2000)
        self.a1 = self.act_fn(fn)

        self.c2 = self.conv_layer(2000, 48)
        self.n2 = self.norm_layer(2048)
        self.a2 = self.act_fn(fn)

        self.c3 = self.conv_layer(2048, 72)
        self.n3 = self.norm_layer(2120)
        self.a3 = self.act_fn(fn)

        self.c4 = self.conv_layer(2120, 96)
        self.n4 = self.norm_layer(2216)
        self.a4 = self.act_fn(fn)

        self.c5 = self.conv_layer(2216, 120)

        self.c6 = self.conv_layer(2336, 1)

    def forward(self, tensor):
        CAT_AXIS = 1
        c1 = self.c1(tensor) # 64, 2000, x0

        c2 = self.n1(c1) # 64, 2000
        c2 = self.a1(c2) # 64, 2000
        c2 = self.c2(c2) # 64, 48, x1

        concat1 = torch.concat([c1, c2], CAT_AXIS) # 2048
        c3 = self.n2(concat1) # 64, 48
        c3 = self.a2(c3) # 64, 48
        c3 = self.c3(c3) # 64, 72, x2

        concat2 = torch.concat([c1, c2, c3], CAT_AXIS) # 2120
        c4 = self.n3(concat2) # 64, 72
        c4 = self.a3(c4) # 64, 72
        c4 = self.c4(c4) # 64, 96, x3

        concat3 = torch.concat([c1, c2, c3, c4], CAT_AXIS) # 2216
        c5 = self.n4(concat3) # 64, 96
        c5 = self.a4(c5) # 64, 96
        c5 = self.c5(c5) # 64, 120, x4

        concat4 = torch.concat([c1, c2, c3, c4, c5], CAT_AXIS) # 2336
        c6 = self.c6(concat4) # 64, 1

        return c6

class ConvNext(nn.Module):
    def conv_layer(self, in_features, out_features):
        if self.dimensions == 2:
            return nn.Conv2d(
                    in_features, 
                    out_features, 
                    kernel_size=7, 
                    padding=PADDING,
                    padding_mode=PADDING_MODE,
                    groups=out_features
            )
        elif self.dimensions == 3:
            return nn.Conv3d(
                    in_features, 
                    out_features, 
                    kernel_size=7, 
                    padding=PADDING,
                    padding_mode=PADDING_MODE,
                    groups=out_features
            )

    def __init__(self, dimensions, n_features, n_hidden_layers, residual_layer):
        super().__init__()
        self.dimensions = dimensions
        self.residual_layer = residual_layer

        self.in_conv = self.conv_layer(1, n_features)
        self.in_act = nn.GELU()

        self.hidden_layers = nn.ModuleList()
        for i in range(n_hidden_layers):
            self.hidden_layers.append(self.conv_layer(n_features, n_features))
            self.hidden_layers.append(self.batch_layer(n_features)) #tf: eps: 0.001, momentum: 0.99
            self.hidden_layers.append(nn.ReLU())

        self.out_conv = self.conv_layer(n_features, 1)
    
    def forward(self, tensor):
        x = self.in_conv(tensor)
        x = self.in_act(x)

        for h_layer in self.hidden_layers:
            x = h_layer(x)

        x = self.out_conv(x)

        if self.residual_layer: x = x + tensor

        return x
