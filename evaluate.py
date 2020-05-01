#############################GENERAL PACKAGES LOADING ########################################################
import argparse
from os import path, listdir, mkdir
import torch

#############################LOAD CUSTOM FUNCTIONS ############################################################
import helper_functions
from helper_functions.imageprocessing import normalize as normalize, reverse
############################# SET PARAMETERS ##################################################################
parser = argparse.ArgumentParser()
# hardware parameters
parser.add_argument('--cuda', action = 'store_true', help= 'use Nvidia GPU or not')

# model parameters
parser.add_argument('--model', default = 'srgan', choices = ['srgan', 'convnet14'], help = 'convnet14|SRGAN')

# data parameters
parser.add_argument('--inputfolder', type = str , default = 'input/', help = ' this speaks for himself')
parser.add_argument('--outputfolder', type = str, default = 'output/', help=' output folder')
parser.add_argument('--nworkers', type=int, default = 2, help = 'numbers of worker for the data loading process')
parser.add_argument('--upSampleFactor', type = int, default = 2, help = 'upscaling factor, default to 2')
parser.add_argument('--n_channels', default = 3, choices = [1,3], help = 'number of input channel for training image, 3 for classic colored image, 1 for black and white')
parser.add_argument('--batch_size', default = 32, type = int, help = ' batch size for training, default to 128')

parser.add_argument('--weight', default = 'generator_final.pth', help = "name of the dictionnary to load")

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

########################## Create the model
if parameters.model in ['srgan']: # adverserial model
    genera = generator(im_channels = parameters.n_channels, upscale_factor = parameters.upSampleFactor).to(device) # no option to change the others parameters yet

########################## Load the data into dataset class and apply random cropping and rotation to generate random highres sample WIP
inputpath = here + '/' + parameters.inputfolder

dataset = {
    'folder' : datasets.ImageFolder(datapath, transform = normalize(parameters.n_channels))
}[parameters.dataset]

######################### Create the dataloaders
dataloader = torch.utils.data.DataLoader(dataset, batch_size=parameters.batch_size,
                shuffle=True, num_workers = parameters.nworkers)

###################### Load the weight
genera.load_state_dict(torch.load(model_path+'weights/'+weightgen))

for parameters in genera.parameters():
    parameters.requires_grad = False
###################### Generate the images and save the image
for n,images in enumerate(dataloader): #  doing it like this, we have the advantage of batch forward operation but we lose the filename info (WIP)
    SRI = genera(images)
    for i in range(parameters.batch_size):
        image = reverse(SRI[i])  # PIL format
        image.save(here+parameters.outputfolder + str(n*parameters.batch_size + i))   #
