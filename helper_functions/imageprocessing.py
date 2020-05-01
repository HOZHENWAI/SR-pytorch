import torchvision.transforms as transforms

### Returns transforms instance

class downsampler:
    """
    to do,
    """
    def __init__(self, lowres_size, n_channels):
        self.n_channels = n_channels
        self.lowres_size = lowres_size
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(self.lowres_size),
                                    transforms.ToTensor()
                                    ])

    def __call__(self, tensorimage):
        out = self.transform(tensorimage)
        return out

def imagegen(highres_size, n_channels):
    """
    To Do,
    """
    out = transforms.Compose([transforms.RandomCrop(highres_size), transforms.RandomRotation(degrees = 30) ,transforms.ToTensor()])
    return out

def normalize(n_channels, statistics):
    """
    To Do,
    """
    mean = {1: statistics[0],
    3 : [statistics[0],statistics[0],statistics[0]]}[n_channels] #unless i get better
    std = {1:statistics[1],
    3:[statistics[1],statistics[1],statistics[1]]}[n_channels] #unless i get better
    out = transforms.Normalize(mean, std)
    return out

def reverse(n_channels, statistics):
    """
    To Do,
    """
    mean = {1: -statistics[0], 3:[-statistics[0],-statistics[0],-statistics[0]]}[n_channels]
    std = {1: 1/statistics[1], 3:[1/statistics[1],1/statistics[1],1/statistics[1]]}[n_channels]
    out =  transforms.Compose([transforms.Normalize(mean,std), transforms.ToPILImage()])
    return out
