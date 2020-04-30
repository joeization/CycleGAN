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
from tqdm import tqdm

from CustomDataset import CustomDataset
from UNet import UNet


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

    G = UNet(3, 3).to(device)

    try:
        G.load_state_dict(torch.load('./genx'))
        print('net loaded')
    except Exception as e:
        print(e)

    plt.ion()
    dataset = 'ukiyoe2photo'

    real_image = 'testB'
    save_image = 'genA'
    save_prefix = './datasets/'+dataset+'/'+save_image+'/'

    image_path_B = './datasets/'+dataset+'/'+real_image+'/*'

    plt.ion()

    train_image_paths_B = glob.glob(image_path_B)
    print(len(train_image_paths_B))

    b_size = 1

    train_dataset_B = CustomDataset(
        train_image_paths_B, train=False)
    train_loader_B = torch.utils.data.DataLoader(
        train_dataset_B, batch_size=b_size, shuffle=False, num_workers=1, pin_memory=False)

    G.eval()

    unloader = transforms.ToPILImage()
    with torch.no_grad():
        loop = tqdm(train_loader_B, desc='inf')
        idx = 1
        for im in loop:
            im = im.to(device)
            gen = G(im)
            gen = (gen+1)/2.0
            im = unloader(gen.squeeze(0).cpu())
            im.save(save_prefix+'%04d.jpg' % idx)
            idx += 1


if __name__ == '__main__':
    run()
