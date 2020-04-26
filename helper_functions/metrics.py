import torch.nn.functional as F
import numpy as np


##################### Peak Signal to Noise ##############
def PSNR(I,I_hat, L):
    """
    To do,
    """
    out = 10*np.log10(L**2 / F.mse_loss(I,I_hat).item())
    return out

class PSNR_metric:
    """
    To do,
    """
    def __init__(self, L = 256):
        self.L = L
    def __call__(self,I, I_hat):
        return PSNR(I, I_hat, self.L)

# Structural similarity
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

class SSIM_metric:
    """
    To do,
    """
    def __init__(self,alpha=0.33, beta=0.33, gamma=0.33,k1=0.1, k2=0.1, k3 = 0.1, L = 256):
        self.C1 = (k1*L)**2
        self.C2 = (k2*L)**2
        self.C3 = (k3*L)**2
    def __call__(self,I, I_hat):
        l = C_luminance(I, I_hat, self.C1)**alpha
        c = C_contrast(I,I_hat, self.C2)**beta
        s = structure_comp(I, I_hat, self.C3)**gamma
        return l*c*s
