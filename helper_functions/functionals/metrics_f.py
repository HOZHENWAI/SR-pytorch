import torch.nn.functional as F
import numpy as np

##################### Peak Signal to Noise ##############
def PSNR(I,I_hat, L):
    """
    To do,
    """
    out = 10*np.log10(L**2 / F.mse_loss(I,I_hat).item())
    return out

##################### Structural similarity
def C_luminance(I, I_hat, C1):
    """
    To do,
    """
    mean_I = torch.mean(I.view(-1)).item()
    mean_I__hat = torch.mean(I.view(-1)).item()
    out = (2*mean_I*mean_I__hat + C1)/(mean_I**2 + mean_I__hat**2 + C1)
    return out
def C_contrast(I, I_hat, C2):
    """
    To do,
    """
    sigma_I = torch.var(I.view(-1)).item()
    sigma_I__hat = torch.var(I_hat.view(-1)).item()
    out = (2*sigma_I * sigma_I__hat + C2)/(sigma_I**2 + sigma_I__hat**2 +C2)
    return out

def structure_comp(I, I_hat, C3):
    """
    To do,
    """
    sigma = np.cov(I.view(-1), I_hat.view(-1))

    #######################redundant part with C_contrast, to do
    sigma_I = torch.var(I.view(-1)).item()
    sigma_I__hat = torch.var(I_hat.view(-1)).item()

    out = (sigma+ C3)/ (sigma + sigma_I__hat + C3)
    return out
