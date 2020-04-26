import torch.nn.functional as F
import torch.nn as nn
import torchvision

# To do :pixel loss
class pixel_loss(nn.Module):
    """
    To do,
    """
    def __init__(self, name = "l1"):
        super().__init__()

    def forward(self, input, target):
        
# to do: content loss

# to do : texture loss

# to do: adversarial loss

# to do: cycle consistency loss

# to do: total variation loss

# to do: prior based loss


# perception loss as defined in 1609.04802v5
class VGG19_loss(nn.Module):
    """
    MSE loss of features extracted from VGG19 network.
    As of now, this loss requires 3 channels colored image as input.
    """
    ###################################### To do, add a RBG-> layers if in_channels = 1

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
    def forward(self, input, target):
        # forward pass the features extractors
        output = self.__getattr__('layer_0')(input)
        for layer in range(1,self.n_layers):
            self.__getattr__('layer_'+ str(layer))(output)
        return self.MSE(output, target)
