import torch.nn as nn




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
