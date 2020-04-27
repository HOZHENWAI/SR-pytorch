import torch.nn as nn
import .helper_functions.functionals as F
import torchvision

# Pixel loss

class charbonnierLoss(nn.Module):
    """
    To do,
    """
    def __init__(self):
        super().__init__()
    def forward(self, input, target):
        F.char_loss(input, target)

class pixel_loss(nn.Module):
    """
    To do,

    Cha = Charbonnier
    """
    def __init__(self, name = "l1"):
        super().__init__()
        assert name in ['l1', 'l2', 'Cha']
        self.loss = {
        'l2' : nn.MSELoss(),
        'l1' : nn.L1Loss(),
        'Cha': nn.charbonnierLoss()
        }[name]
    def forward(self, input, target):
        return self.loss(input,target)

# Texture loss

# to do: adversarial loss

# to do: cycle consistency loss

# to do: total variation loss

# to do: prior based loss


################################ Content loss: content as defined as


# features extractor
class VGG19_features_extractor(nn.Module):
    """
    To do,
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
    def forward(self, input):
        output = self.__getattr__('layer_0')(input)
        for layer in range(1,self.n_layers):
            output = self.__getattr__('layer_'+ str(layer))(output)
        return output
# perception loss as defined in 1609.04802v5 (features of VGG)
class VGG19_loss(nn.Module):
    """
    MSE loss of features extracted from VGG19 network.
    As of now, this loss requires 3 channels colored image as input.
    """
    ###################################### To do, add a RBG-> layers if in_channels = 1

    def __init__(self, conv_num, max_pool_layer):
        super().__init__()
        self.fextract = VGG19_features_extractor(conv_num, max_pool_layer)
        self.MSE = nn.MSELoss()
    def forward(self, input, target):
        # forward pass the features extractors
        return self.MSE(self.fextract(input), self.fextract(target))
