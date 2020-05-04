#############################GENERAL PACKAGES LOADING ########################################################
import argparse
from os import path, listdir, mkdir
import torch
from torch import optim
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter

#############################LOAD CUSTOM FUNCTIONS ############################################################
import helper_functions
import helper_functions.imageprocessing.imagegen as imagegen
import helper_functions.utils.ModelTrainer as ModelTrainer
############################# SET PARAMETERS ##################################################################
parser = argparse.ArgumentParser()
# hardware parameters
parser.add_argument('--cuda', action = 'store_true', help= 'use Nvidia GPU or not')

# model parameters
parser.add_argument('--model', default = 'srgan', choices = ['srgan'], help = 'SRGAN')

# data parameters
parser.add_argument('--datafolder', type = str , default = 'data/', help = ' this speaks for himself')
parser.add_argument('--dataset' , default = 'folder', choices = ['folder'], help= 'folder')
parser.add_argument('--nworkers', type=int, default = 2, help = 'numbers of worker for the data loading process')
parser.add_argument('--imagesize', default = (64,64), help = 'dimension of the training high dim images')
parser.add_argument('--upSampleFactor', type = int, default = 2, help = 'upscaling factor, default to 2')
parser.add_argument('--n_channels', default = 3, choices = [1,3], help = 'number of input channel for training image, 3 for classic colored image, 1 for black and white')

# loss parameters
parser.add_argument('--losscontent', default = 'MSE', choices = ['MSE', 'VGG19'], help = 'loss function, default to pixel wise MSE loss')
parser.add_argument('--lossadv', default = 'BCE', choices= ['BCE'], help = 'adverserial loss')

# optimization parameters
parser.add_argument('--batch_size', default = 32, type = int, help = ' batch size for training, default to 128')
parser.add_argument('--epochs', default = 100, type = int, help = ' number of epochs, default to 100')
parser.add_argument('--optimizer', default = 'Adam', choices = ['Adam'], help='optimizer, default to Adam')
parser.add_argument('--lrate', default = 0.0002, type=float, help = 'learning rate for optimizer')
parser.add_argument('--weightgen', default = '', help = 'generator weights name')
parser.add_argument('--weightdis', default = '', help = 'discrimimator weights name')
parser.add_argument('--weightpath', default = 'weight/', help ='')
parameters = parser.parse_args()

############################################## END SET PARAMETERS #################################################

############################ initialize cuda device
if parameters.cuda == True:
    assert torch.cuda.is_available()
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

###########################get current dir
here = path.abspath(path.dirname(__file__))
model_path = here +'/'+ parameters.model+'/'

if parameters.weightpath:
    weight_path = model_path + parameters.weight_path
else:
    weight_path = ''
########################## Create the models

if parameters.model in ['srgan']: # adverserial model
    genera = generator(im_channels = parameters.n_channels, upscale_factor = parameters.upSampleFactor) # no option to change the others parameters yet
    discri = discriminator(im_channels = parameters.n_channels, highres_size = parameters.imagesize) # no option to change the others parameters yet

##########################Load the weights of the pretrained models
genera.load_weight(weight= parameters.weightgen, weight_path= weight_path)
if discri:
    discri.load_weight(weight = parameters.weightpath, weight_path = weight_path)

########################## Load the data into dataset class and apply random cropping and rotation to generate random highres sample WIP
datapath = here + '/' + parameters.datafolder

dataset = {
    'folder' : datasets.ImageFolder(datapath, transform = imagegen(parameters.imagesize, parameters.n_channels))
}[parameters.dataset]

######################### Create the dataloaders
dataloader = torch.utils.data.DataLoader(dataset, batch_size=parameters.batch_size,
                shuffle=True, num_workers = parameters.nworkers)

######################### Create the lowres image generator
lowres_size = (parameters.imagesize[0]//parameters.upSampleFactor, parameters.imagesize[1]//parameters.upSampleFactor)
sampler = helper_functions.imageprocessing.downsampler(lowres_size, n_channels = parameters.n_channels)
# normalizer = helper_functions.imageprocessing.normalize(n_channels = parameters.n_channels)


######################## Define the losses
loss_content = {
    'MSE' : nn.MSELoss(),
    'VGG' : helper_functions.loss.VGG19_loss(conv_num = 4, max_pool_layer = 4) # as defined in the SRGAN paper
}[parameters.losscontent]

loss_adv = {
    'BCE' : nn.BCELoss()
}[parameters.lossadv]

######################## Set the training logger
writer = SummaryWriter(here+'/logs')

####################### Create training instance
train_instance = ModelTrainer(parameters.model, genera, loss_content, dataloader, parameters.optimizer, sampler, discri, loss_adv)

###################### Finally the training phase
train_instance.train(parameters.batch_size,
    parameters.epochs, parameters.lrate, device, metrics= 'PSNR', board = writer, save_loc = weight_path)
