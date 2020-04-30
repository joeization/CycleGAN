import glob
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as tdist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
# from skimage import feature

# from CustomDataset import CustomDataset
# from discriminator_v2 import Discriminator, Discriminator_S1, Discriminator_S2
# from generator_v2 import Generator_in, Generator, Generator2, GaussianNoise
from generator_v3 import Generator_v3
# from generator_v2_no_gau import Generator_in, Generator, Generator2
# from utility import center_crop


def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


def run():
    random.seed()
    np.random.seed()
    torch.multiprocessing.freeze_support()
    print('loop')
    # torch.backends.cudnn.enabled = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # GaussianNoise.device = device
    # device = torch.device("cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    latent_s = 32
    G = Generator_v3(latent_length=latent_s,
                     hidden_length=2048).to(device)
    try:
        G.load_state_dict(torch.load('./gen'))
        # G.load_state_dict(torch.load('./V2_total_loss/gen'))
        # D.load_state_dict(torch.load('./V2_total_loss/fcn2'))
        print('net loaded')
    except Exception as e:
        print(e)

    plt.ion()

    G.eval()
    # ran = torch.autograd.Variable(torch.Tensor(
    #     np.random.normal(0, 1, (1, 128)))).to(device)
    unloader = transforms.ToPILImage()
    with torch.no_grad():
        for i in range(50):
            print(i)
            z = torch.autograd.Variable(torch.Tensor(
                np.random.normal(0, 1, (1, latent_s)))).to(device)
            gen, pz = G(z.detach())
            z.requires_grad = False
            gen = (gen+1)/2.0
            im = unloader(gen.squeeze(0).cpu())
            im.save('./generate/%02d.png' % i)
            # im.save('./tb/'+str(i)+'ntb.png')


if __name__ == '__main__':
    run()
