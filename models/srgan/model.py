from models.modules import residual_block1lrelu, residual_block2prelu, upscaler_block
import torch
import torch.nn as nn


class generator(nn.Module):
    """
    Generator network
    """
    def __init__(self, im_channels = 3,n_residual_blocks = 5, upscale_factor = 4, n_neurons_p1 = 64):
        super().__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upscale_factor = upscale_factor
        self.im_channels = im_channels
        self.n_neurons_p1 = n_neurons_p1

        self.conv1 = nn.Conv2d(in_channels = self.im_channels, out_channels = self.n_neurons_p1, kernel_size = 9, stride = 1, padding = 4)
        self.Prelu = nn.PReLU()

        for n in range(self.n_residual_blocks):
            self.add_module('residual_block_'+str(n), residual_block2prelu(in_channels = self.n_neurons_p1, neurons = self.n_neurons_p1))

        self.conv2 = nn.Conv2d(in_channels = self.n_neurons_p1, out_channels=self.n_neurons_p1, stride = 1, kernel_size = 3, padding = 1)
        self.BN = nn.BatchNorm2d(self.n_neurons_p1)
        for u in range(self.upscale_factor//2):
            self.add_module('upscale_block_'+str(u), upscaler_block(in_channels = self.n_neurons_p1 , neurons = self.n_neurons_p1*4))

        self.conv3 = nn.Conv2d(in_channels = self.n_neurons_p1, out_channels = self.im_channels, kernel_size = 9, stride = 1, padding = 4)

    def load_weight(self,weight= None, weight_path=''):
        """
        load weight from location
        """
        if weight:
            self.load_state_dict(torch.load(weight_path+weight))

    def forward(self,input):
        output = self.conv1(input)
        output = self.Prelu(output)
        skip = output.clone()

        for n in range(self.n_residual_blocks):
            output = self.__getattr__('residual_block_'+str(n))(output)

        output = self.conv2(output)
        output = self.BN(output)
        output += skip

        for u in range(self.upscale_factor//2):
            output = self.__getattr__('upscale_block_'+str(u))(output)

        output = self.conv3(output)

        return output

class discriminator(nn.Module):
    """
    Discrimimation network
    """
    def __init__(self, n_residual_blocks = 7, im_channels = 3, n_neurons_p1 = 64 , n_neurons_p2 = 1024, highres_size=(64,64)):
        super().__init__()
        self.n_residual_blocks = n_residual_blocks
        self.im_channels = im_channels
        self.n_neurons_p1 = n_neurons_p1
        self.n_neurons_p2 = n_neurons_p2

        self.conv1 = nn.Conv2d(in_channels = self.im_channels, out_channels = self.n_neurons_p1, kernel_size = 3, stride = 1, padding = 1)
        self.Lrelu1 = nn.LeakyReLU()
        self.add_module('residual_block_0', residual_block1lrelu(in_channels = self.n_neurons_p1, neurons = self.n_neurons_p1, stride = 2))

        size = highres_size

        input_channel = self.n_neurons_p1
        for n in range(1,self.n_residual_blocks):
            multiplier = 2**(n//2 +1)
            stride = 1+ (n+1)%2
            self.add_module('residual_block_'+str(n), residual_block1lrelu(in_channels = input_channel, neurons = self.n_neurons_p1*multiplier, stride = stride))
            input_channel = self.n_neurons_p1*multiplier
            size = (size[0] // stride, size[1]//stride)

        self.flat = nn.Flatten()
        self.dense1 = nn.Linear(in_features = input_channel*size[0]*size[1]//4, out_features = self.n_neurons_p2)
        self.Lrelu2 = nn.LeakyReLU()
        self.dense2 = nn.Linear(in_features = self.n_neurons_p2, out_features = 1)

    def load_weight(self,weight= None, weight_path=''):
        """
        load weight from location
        """
        if weight:
            self.load_state_dict(torch.load(weight_path+weight))

    def forward(self, input):
        output = self.conv1(input)
        output = self.Lrelu1(output)
        for n in range(self.n_residual_blocks):
            output = self.__getattr__('residual_block_'+str(n))(output)

        output = self.flat(output)
        output = self.dense1(output)
        output = self.Lrelu2(output)
        output = self.dense2(output)
        output = torch.sigmoid(output)
        return output
