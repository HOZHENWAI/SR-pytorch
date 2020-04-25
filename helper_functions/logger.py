from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class Visualizer:
    """
    To do,
    """
    def __init__(self, n_channels = 3, image_size = (64,64)):
        mean = {1: -0.5, 3:[-0.5,-0.5,-0.5]}[n_channels] #################### To be Removed later
        std = {1: 1/0.5, 3:[1/0.5,1/0.5,1/0.5]}[n_channels]  ################## Same
        self.transform =  transforms.Compose([transforms.Normalize(mean,std), transforms.ToPILImage()])# HARD CODED FOR NOW
        self.figure, (self.axlr, self.axhr_real, self.axhr_fake) = plt.subplots(1,3)
        self.figure.show()

    def __call__(self,lr,hr_real, hr_fake):
        lr_image = self.transform(lr)
        hr_realimage = self.transform(hr_real)
        hr_fakeimage = self.transform(hr_fake)

        self.axlr.imshow(lr_image)
        self.axlr.imshow(hr_realimage)
        self.axlr.imshow(hr_fakeimage)

        return self.figure
