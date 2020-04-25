import torch.optim as optim
import torch
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

    def train(self,batch_size, epochs, lrate, lowres_size, device,metrics = 'PSNR',,tensorboard = False):
        """

        """
        optim_generator = {
            'Adam' : optim.Adam(self.generator.parameters(), lr = lrate)
        }[self.optimizer]

        if self.discriminator:
            optim_discriminator ={
            'Adam' : optim.Adam(self.discriminator.parameters(), lr = lrate)
            }[self.optimizer]

        # Get epochs losses (take the mean here)
        G_losses = []
        if self.discriminator:
            D_losses = []


        # Create a tensor to hold the low res images for each batch
        low_res_fake = torch.FloatTensor(batch_size,lowres_size[0], lowres_size[1],lowres_size[2] ).to(device)

        # iterate through all epochs
        for epoch in range(epochs):

            # track it iterations numbers for visualisation purposes
            iters = 0

            # track the mean loss through the epochs
            G_losses_mean = 0
            if self.discriminator:
                D_losses_mean = 0

            for i, images in enumerate(self.dataloader):
                iters += 1
                # load n_batch images
                high_res_real, _ = images
                for j in range(batch_size):
                    low_res_fake[j] = self.sampler(high_res_real[j]).to(device)
                    high_res_real[j] = self.normalizer(high_res_real[j]).to(device)

                real_label = torch.full((batch_size,), 1, device = device)
                fake_label = torch.full((batch_size,), 0, device = device)
                #create fake outputs
                high_res_fake = self.generator(low_res_fake).to(device)

                ## Training the discriminator
                if discriminator:
                    ## Train with all real batch_size

                    self.discriminator.zero_grad()
                    discriminator_real = self.loss_adv(self.discriminator(high_res_real), real_label).to(device)
                    discriminator_real.backward()
                    discriminator_fake = self.loss_adv(self.discriminator(high_res_fake), fake_label).to(device)
                    discriminator_fake.backward()
                    discriminator_loss = discriminator_real + discriminator_fake

                    D_losses_mean += discriminator_loss.data[0]

                    optim_discriminator.step()

                ## Training the Generator
                self.generator.zero_grad()
                generator_loss = self.loss_content(high_res_fake, high_res_real).to(device)
                G_losses_mean += generator_loss.data[0]

                generator_loss.backward()
                optim_generator.step()

            # End of epochs
            G_losses_mean = G_losses_mean / len(self.dataloader)
            if self.discriminator:
                D_losses_mean = D_losses_mean / len(self.dataloader)
