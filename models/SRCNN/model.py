from torch.nn import Module as
from models.modules import upscaler_preprocessing, tripleconvrelu



# models as defined in 1501.00092

class srcnn(Module):
    """
    To do, need to add flexibility to this model
    """
    def __init__(self, im_channels = 3, upscale_factor = 2):
        super().__init__()
        self.im_channels = im_channels
        self.upscale_factor = upscale_factor(in_channels=self.im_channels)
        self.upsample = upscaler_preprocessing(kernels = im_channels, scale_factor = upscale_factor)
        self.tripleconv1 = tripleconvrelu(in_channels = self.im_channels, n_filters = [64,32,self.im_channels], kernels = [9,1,5])

    def load_weight(self,weight= None, weight_path=''):
        """
        load weight from location
        """
        if weight:
            self.load_state_dict(torch.load(weight_path+weight))

    def forward(self, input):
        """

        """
        output = self.upscale_factor(input)
        output = self.tripleconv1(output)
        return output
