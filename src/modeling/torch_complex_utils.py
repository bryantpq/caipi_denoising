import numpy as np
import torch
import pdb

from math import pi

def complex_mse(output, target):
    #loss = torch.mean((output - target)**2)
    #loss = torch.abs(loss)

    loss = ( 0.5 * (output - target)**2 ).mean(dtype=torch.complex64)
    return torch.abs(loss)

class ComplexConv2d(torch.nn.Module):
    def __init__(self, in_features, out_features, kernel_size, padding, padding_mode):
        super().__init__()

        if in_features  != 1: in_features  = in_features  // 2
        if out_features != 1: out_features = out_features // 2

        self.real_real_conv = torch.nn.Conv2d(
                in_features,
                out_features,
                (kernel_size, kernel_size),
                stride=1,
                padding=padding,
                padding_mode=padding_mode
        )
        self.imag_real_conv = torch.nn.Conv2d(
                in_features,
                out_features,
                (kernel_size, kernel_size),
                stride=1,
                padding=padding,
                padding_mode=padding_mode
        )
        self.real_imag_conv = torch.nn.Conv2d(
                in_features,
                out_features,
                (kernel_size, kernel_size),
                stride=1,
                padding=padding,
                padding_mode=padding_mode
        )
        self.imag_imag_conv = torch.nn.Conv2d(
                in_features,
                out_features,
                (kernel_size, kernel_size),
                stride=1,
                padding=padding,
                padding_mode=padding_mode
        )

    def forward(self, x):
        torch_real, torch_imag = torch.real(x), torch.imag(x)

        torch_real_real = self.real_real_conv(torch_real)
        torch_imag_real = self.imag_real_conv(torch_imag)
        torch_real_imag = self.real_imag_conv(torch_real)
        torch_imag_imag = self.imag_imag_conv(torch_imag)

        real_out = torch_real_real - torch_imag_imag
        imag_out = torch_imag_real + torch_real_imag
        torch_output = torch.complex(real_out, imag_out)

        return torch_output

class ComplexConv3d(torch.nn.Module):
    def __init__(self, in_features, out_features, kernel_size, padding, padding_mode):
        super().__init__()
        if in_features  != 1: in_features  = in_features // 2
        if out_features != 1: out_features = out_features // 2

        self.real_real_conv = torch.nn.Conv3d(
                in_features,
                out_features,
                (kernel_size, kernel_size, kernel_size),
                stride=1,
                padding=padding,
                padding_mode=padding_mode
        )
        self.imag_real_conv = torch.nn.Conv3d(
                in_features,
                out_features,
                (kernel_size, kernel_size, kernel_size),
                stride=1,
                padding=padding,
                padding_mode=padding_mode
        )
        self.real_imag_conv = torch.nn.Conv3d(
                in_features,
                out_features,
                (kernel_size, kernel_size, kernel_size),
                stride=1,
                padding=padding,
                padding_mode=padding_mode
        )
        self.imag_imag_conv = torch.nn.Conv3d(
                in_features,
                out_features,
                (kernel_size, kernel_size, kernel_size),
                stride=1,
                padding=padding,
                padding_mode=padding_mode
        )

    def forward(self, x):
        torch_real, torch_imag = torch.real(x), torch.imag(x)

        torch_real_real = self.real_real_conv(torch_real)
        torch_imag_real = self.imag_real_conv(torch_imag)
        torch_real_imag = self.real_imag_conv(torch_real)
        torch_imag_imag = self.imag_imag_conv(torch_imag)

        real_out = torch_real_real - torch_imag_imag
        imag_out = torch_imag_real + torch_real_imag
        torch_output = torch.complex(real_out, imag_out)

        return torch_output

class CReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        real_out, imag_out = torch.real(x), torch.imag(x)
        real_out = torch.nn.ReLU()(real_out)
        imag_out = torch.nn.ReLU()(imag_out)

        torch_output = torch.complex(real_out, imag_out)

        return torch_output

class CGELU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        real_out, imag_out = torch.real(x), torch.imag(x)
        real_out = torch.nn.GELU()(real_out)
        imag_out = torch.nn.GELU()(imag_out)

        torch_output = torch.complex(real_out, imag_out)

        return torch_output
