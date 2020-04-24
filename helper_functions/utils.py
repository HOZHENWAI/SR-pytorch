import torch.optim as optim
# place visualisation function here
class ModelTrainer:
    """
    Some wrapper class to train all future model
    """
    def __init__(self, generator, loss, datawrapper, batch_size, epochs, optimizer, lrate, weightgen = None, discriminator = None, loss_adv = None, weigthdis = None):
        # copy the parameters
        self.generator = generator
        self.loss = loss
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lrate = lrate
        self.weightgen = weightgen
        self.discriminator = discriminator
        self.loss_adv = loss_adv
        self.weightdis = weigthdis

        # load models weights as needed
        if parameters.weightgen:
            generator.load_state_dict(torch.load(model_path+'weights/'+parameters.weightgen))

        if parameters.model in ['srgan']:
            # making sure in that case that a discriminator is given
            assert self.discriminator
            assert self.loss_adv
            if parameters.weightdis:
                discriminator.load_state_dict(torch.load(model_path+'weights/'+parameters.weightdis))
