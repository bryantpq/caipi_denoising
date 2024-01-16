import numpy as np
import pdb
import torch
import torch.nn as nn

from modeling.torch_complex_utils import ComplexConv2d, ComplexConv3d, CReLU

RESIDUAL_LAYER = True
N_HIDDEN_LAYERS = 15

def get_model(model_type, dimensions):
    if model_type == 'dncnn':
        model = DnCNN(dimensions, 64, N_HIDDEN_LAYERS)
    elif model_type == 'cdncnn':
        model = CDnCNN(dimensions, 128, N_HIDDEN_LAYERS)

    return model

class DnCNN(nn.Module):
    def conv_layer(self, in_features, out_features):
        if self.dimensions == 2:
            return nn.Conv2d(
                    in_features, 
                    out_features, 
                    (3, 3),
                    stride=1,
                    padding='same'
            )
        elif self.dimensions == 3:
            return nn.Conv3d(
                    in_features, 
                    out_features, 
                    (3, 3, 3),
                    stride=1,
                    padding='same'
            )

    def batch_layer(self, out_features):
        if self.dimensions == 2:
            return nn.BatchNorm2d(out_features)
        elif self.dimensions == 3:
            return nn.BatchNorm3d(out_features)

    def __init__(self, dimensions, n_features, n_hidden_layers, residual_layer=RESIDUAL_LAYER):
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
            return ComplexConv2d(in_features, out_features, 3)
        elif self.dimensions == 3:
            return ComplexConv3d(in_features, out_features, 3)

    def __init__(self, dimensions, n_features, n_hidden_layers, residual_layer=RESIDUAL_LAYER):
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
