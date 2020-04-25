import torchvision.transforms as transforms


def downsampler(lowres_size,n_channels):
    """
    To Do,
    Prepare all necessary steps to generate a low image from a high res image in tensor format.
    """
    mean = {1: 0.5,
    3 : [0.5,0.5,0.5]}[n_channels] #unless i get better
    std = {1:0.5,
    3:[0.5,0.5,0.5]}[n_channels] #unless i get better
    out = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(lowres_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)
                                ])
    return out

def imagegen(highres_size, n_channels):
    """
    To Do,
    """
    out = transforms.Compose([transforms.RandomCrop(highres_size), transforms.RandomRotation(degrees = 30) ,transforms.ToTensor()])
    return out

def normalize(n_channels):
    """
    To Do,
    """
    mean = {1: 0.5,
    3 : [0.5,0.5,0.5]}[n_channels] #unless i get better
    std = {1:0.5,
    3:[0.5,0.5,0.5]}[n_channels] #unless i get better
    out = transforms.Normalize(mean, std)
    return out

def reverse(n_channels):
    """
    To Do,
    """
    mean = {1: -0.5, 3:[-0.5,-0.5,-0.5]}[n_channels]
    std = {1: 1/0.5, 3:[1/0.5,1/0.5,1/0.5]}[n_channels]
    out =  transforms.Compose([transforms.Normalize(mean,std), transforms.ToPILImage()])
    return out
