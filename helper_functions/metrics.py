import .helper_functions.functionals as F
##################################### To Do: metrics should have require_grad = False


##################### Peak Signal to Noise ##############
class PSNR_metric:
    """
    To do,
    """
    def __init__(self, L = 256):
        self.L = L
    def __call__(self,I, I_hat):
        return F.PSNR(I, I_hat, self.L)

# Structural similarity


class SSIM_metric:
    """
    To do,
    """
    def __init__(self,alpha=0.33, beta=0.33, gamma=0.33,k1=0.1, k2=0.1, k3 = 0.1, L = 256):
        self.C1 = (k1*L)**2
        self.C2 = (k2*L)**2
        self.C3 = (k3*L)**2
    def __call__(self,I, I_hat):
        l = F.C_luminance(I, I_hat, self.C1)**alpha
        c = F.C_contrast(I,I_hat, self.C2)**beta
        s = F.structure_comp(I, I_hat, self.C3)**gamma
        return l*c*s
