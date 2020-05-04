import torch.nn as nn


class upscaler_preprocessing(nn.Module):
    """

    """
    def __init__(self, in_channels = 3, scale_factor = 2):
        super().__init__()
        self.mode = {
        3 : 'bicubic',
        2 : 'bilinear'
        }[in_channels]
        self.upsampler = nn.Upsample(scale_factor = scale_factor, mode = self.mode)
        for parameters in self.parameters():
            parameters.requires_grad = False

    def forward(self, input):
        output = self.upsampler(input)
        return output


class tripleconvrelu(nn.Module):
    """to do

    """

    def __init__(self, in_channels = 3, n_filters = [64,32,3], kernels = [9,1,5]):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = n_filters[0], kernel_size = kernels[0], padding = (kernels[0]-1)//2)
        self.re1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels = n_filters[0], out_channels = n_filters[1], kernel_size = kernels[1], padding = (kernels[1]-1)//2)
        self.re2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels = n_filters[1], out_channels = n_filters[2], kernel_size = kernels[2], padding = (kernels[2]-1)//2)
    def forward(self, input):
        output = self.conv1(input)
        output = self.re1(output)
        output = self.conv2(output)
        output = self.re2(output)
        output = self.conv3(output)
        return output

class residual_block2prelu(nn.Module):
    """
    To do

    """
    def __init__(self, in_channels = 64,kernels = 3, neurons = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = neurons, kernel_size = kernels, stride = 1, padding = (kernels-1)//2)
        self.BN = nn.BatchNorm2d(num_features = neurons)
        self.Prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels = neurons, out_channels = neurons, kernel_size = kernels, stride = 1, padding = (kernels-1)//2)
        self.BN = nn.BatchNorm2d(num_features = neurons)

    def forward(self, input):
        output = self.conv1(input)
        output = self.BN(output)
        output = self.Prelu(output)
        output = self.conv2(output)
        output = self.BN(output)
        output += input
        return output

class upscaler_block(nn.Module):
    """
    To do

    """
    def __init__(self, in_channels =64, kernels = 3, neurons = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = neurons, kernel_size = kernels, stride = 1, padding = (kernels-1)//2)
        self.Pixel1 = nn.PixelShuffle(upscale_factor = 2)
        self.Prelu = nn.PReLU()
    def forward(self, input):
        output = self.conv1(input)
        output = self.Pixel1(output)
        output = self.Prelu(output)
        return output

class residual_block1lrelu(nn.Module):
    """
    To do
    """
    def __init__(self, in_channels = 64, kernels = 3, neurons = 64, stride = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = neurons, kernel_size = kernels, stride = stride, padding = (kernels-1)//2)
        self.BN = nn.BatchNorm2d(num_features = neurons)
        self.LRelu = nn.LeakyReLU()
    def forward(self, input):
        output = self.conv1(input)
        output = self.BN(output)
        output = self.LRelu(output)
        return output
