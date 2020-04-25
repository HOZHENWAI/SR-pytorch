import torch.nn.functional as F
import torch.nn as nn
import torchvision

# pixel loss


# content loss

# texture loss

# adversarial loss

# cycle consistency loss

# total variation loss

# prior based loss


# perception loss as defined in 1609.04802v5
class VGG19_loss(nn.Module):
    """
    MSE loss of features extracted from VGG19 network
    """
    def __init__(self, conv_num, max_pool_layer):
        super().__init__()
        vgg = torchvision.models.vgg19(pretrained=True)
        self.n_layers = 2*conv_num + {
        1: 0,
        2: 5,
        3: 10,
        4: 28,
        }[max_pool_layer]
        # extract the features
        for layer in range(self.n_layers):
            self.add_module('layer_'+str(layer), vgg.features[layer])
        # fix the features
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.MSE = nn.MSELoss()
    def forward(self, input):
        # forward pass the features extractors
        output = self.__getattr__('layer_0')(input)
        for layer in range(1,self.n_layers):
            self.__getattr__('layer_'+ str(layer))(output)
        return self.MSE(output)
