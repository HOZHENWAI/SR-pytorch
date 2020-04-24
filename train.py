import argparse
from os import path, listdir, mkdir
import torch
from torch import optim
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms

import helper_functions
import helper_functions.utils.ModelTrainer as ModelTrainer
############################# SET PARAMETERS
parser = argparse.ArgumentParser()
# hardware parameters
parser.add_argument('--cuda', action = 'store_true', help= 'use Nvidia GPU or not')

# model parameters
parser.add_argument('--model', default = 'srgan', choice = ['srgan', 'convnet14'], help = 'convnet14|SRGAN')

# data parameters
parser.add_argument('--datafolder', type = str , default = 'data/', help = ' this speaks for himself')
parser.add_argument('--dataset' , default = 'folder', choice = ['folder'], help= 'folder')
parser.add_argument('--nworkers', type=int, default = 2, help = 'numbers of worker for the data loading process')
parser.add_argument('--upSampleFactor', type = int, default = 2, help = 'upscaling factor, default to 2')
parser.add_argument('--n_channels', default = 3, choice = [1,3], help = 'number of input channel for training image, 3 for classic colored image, 1 for black and white')

# loss parameters
parser.add_argument('--losscontent', default = 'MSE', choice = ['MSE', 'VGG19'], help = 'loss function, default to pixel wise MSE loss')
parser.add_argument('--lossadv', default = 'BCE', choice= ['BCE'], help = 'adverserial loss')

# optimization parameters
parser.add_argument('--batch_size', default = 128, type = int, help = ' batch size for training, default to 128')
parser.add_argument('--epochs', default = 100, type = int, help = ' number of epochs, default to 100')
parser.add_argument('--optimizer', default = 'Adam', choice = ['Adam'], help='optimizer, default to Adam')
parser.add_argument('--lrate', default = 0.0002, type=float, help = 'learning rate for optimizer')
parser.add_argument('--weightgen', default = '', help = 'generator weights name')
parser.add_argument('--weightdis', default = '', help = 'discrimimator weights name')

parameters = parser.parse_args()

############################## END SET PARAMETERS

# initialise cuda device
if parameters.cuda == True:
    assert torch.cuda.is_available()
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

#get current dir
here = path.abspath(path.dirname(__file__))
model_path = here +'/'+ parameters.model+'/'

# Create the model
if parameters.model in ['srgan']: # adverserial model
    generator = generator(im_channels = parameters.n_channels).to(device) # no option to change the others parameters yet
    discriminator = discriminator(im_channels = parameters.n_channels).to(device) # no option to change the others parameters yet


# Load the data into dataset class
datapath = here + '/' + parameters.datafolder

dataset = {
    'folder' : datasets.ImageFolder(datapath, transform = transforms.ToTensor())
}[parameters.dataset]

# Create the dataloaders
dataloader = torch.utils.data.DataLoader(dataset, batch_size=parameters.batch_size,
                shuffle=True, num_workers = parameters.nworkers)

# Define the loss
loss_content = {
    'MSE' : nn.MSELoss(),
    'VGG' : helper_functions.loss.VGG19_loss(conv_num = 5, max_pool_layer = 4) # as defined in the SRGAN paper
}[parameters.losscontent]

loss_adv = {
    'BCE' : nn.BCELoss()
}[parameters.lossadv]

# Create training instance

train_instance = ModelTrainer(generator, loss_content, dataloader, parameters.batch_size,
                    parameters.epochs, parameters.optimizer, parameters. lrate,
                    weightgen = parameters.weightgen, discriminator = discriminator, loss_adv = loss_adv,
                    weigthdis = parameters.weightdis)
