import torch.optim as optim
import torch
from torch.autograd import Variable

from helper_functions.logger import Visualizer
# place visualisation function here
class ModelTrainer:
    """
    Some wrapper class to train all future model
    """
    def __init__(self, model,generator, loss_content, dataloader, optimizer, sampler, normalizer,discriminator = None, loss_adv = None):
        # copy the parameters
        self.model = model
        self.generator = generator
        self.loss_content = loss_content
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.discriminator = discriminator
        self.loss_adv = loss_adv
        self.sampler = sampler
        self.normalizer = normalizer
    def load_weight(self,weightgen = None, weightdis = None):
        """
        load weight from location
        """
        if weightgen:
            self.generator.load_state_dict(torch.load(model_path+'weights/'+weightgen))

        if parameters.model in ['srgan']:
            # making sure in that case that a discriminator is given
            assert self.discriminator
            assert self.loss_adv
            if parameters.weightdis:
                self.discriminator.load_state_dict(torch.load(model_path+'weights/'+weightdis))

    def train(self,batch_size, epochs, lrate, lowres_size, device, save_loc, metrics = 'PSNR',board = None):
        """
            To do
        """
        # send the models to device
        self.generator.to(device)
        self.loss_content.to(device)
        if self.discriminator:
            self.discriminator.to(device)
            self.loss_adv.to(device)

        # create the optimizer
        optim_generator = {
            'Adam' : optim.Adam(self.generator.parameters(), lr = lrate)
        }[self.optimizer]

        if self.discriminator:
            optim_discriminator ={
            'Adam' : optim.Adam(self.discriminator.parameters(), lr = lrate)
            }[self.optimizer]

        ########### get the logger running: get the models first
        if board:
            # board.add_graph(self.generator)
            ImageV = Visualizer(lowres_size[0], (lowres_size[1], lowres_size[2]))




        # iterate through all epochs
        for epoch in range(epochs):

            ############### track it iterations numbers for visualisation purposes
            iters = 0

            ############### track the mean loss through the epochs
            G_losses_mean = 0
            if self.discriminator:
                D_losses_mean = 0

            for i, images in enumerate(self.dataloader):
                iters += 1
                ############# load n_batch images
                high_res_real, _ = images

                ########### Create a tensor to hold the low res images for each batch
                low_res = torch.FloatTensor(high_res_real.size(0),lowres_size[0], lowres_size[1],lowres_size[2], device = device )


                for j in range(high_res_real.size(0)):
                    low_res[j] = self.sampler(high_res_real[j])
                    high_res_real[j] = self.normalizer(high_res_real[j])

                real_label = torch.full((high_res_real.size(0),1), 1, device = device)
                fake_label = torch.full((high_res_real.size(0),1), 0, device = device)
                #create fake outputs
                high_res_fake = self.generator(low_res.to(device))
                high_res_real = high_res_real.to(device)
                ######### if tensorboard, visualise some images
                if board:
                    board.add_figure('LRvsHRvsHRFake', ImageV(low_res[0], high_res_real[0], high_res_fake[0]),epoch*len(self.dataloader)+iters)

                ######### Training the discriminator
                if self.discriminator:
                    ####### Train with all real batch_size

                    self.discriminator.zero_grad()
                    discriminator_real = self.loss_adv(self.discriminator(high_res_real), real_label).to(device)
                    discriminator_real.backward()
                    discriminator_fake = self.loss_adv(self.discriminator(high_res_fake.detach()), fake_label).to(device)
                    discriminator_fake.backward()
                    discriminator_loss = discriminator_real + discriminator_fake

                    D_losses_mean += discriminator_loss.item()

                    optim_discriminator.step()

                ######### Training the Generator
                self.generator.zero_grad()
                generator_loss = self.loss_content(high_res_fake, high_res_real).to(device)
                G_losses_mean += generator_loss.item()

                generator_loss.backward()
                optim_generator.step()

            ########## End of epoch
            G_losses_mean = G_losses_mean / len(self.dataloader)
            if board:
                board.add_scalar('generator_loss', G_losses_mean, epoch)

            if self.discriminator:
                D_losses_mean = D_losses_mean / len(self.dataloader)
                if board:
                    board.add_scalar('discriminator_loss', D_losses_mean, epoch)
            torch.save(self.generator.state_dict(), save_loc+'weights/generator_final.pth')
            if self.discriminator:
                torch.save(self.discriminator.state_dict(), save_loc+'weights/discriminator_final.pth')
