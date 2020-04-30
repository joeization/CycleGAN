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
from skimage import feature
from tqdm import trange, tqdm

from CustomDataset import CustomDataset
from discriminator import Discriminator
from UNet import UNet
from utility import center_crop


def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


class cropper(nn.Module):
    def __init__(self, img_size=64, crop_size=32):
        super(cropper, self).__init__()
        self.isz = img_size
        self.csz = crop_size

    def forward(self, x):
        sx = random.randint(0, self.isz-1-self.csz)
        sy = random.randint(0, self.isz-1-self.csz)
        return x[:, :, sx:sx+self.csz, sy:sy+self.csz]


def run():
    print('loop')
    # torch.backends.cudnn.enabled = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    Dx = Discriminator().to(device)
    Gx = UNet(3, 3).to(device)

    Dy = Discriminator().to(device)
    Gy = UNet(3, 3).to(device)

    ld = False
    if ld:
        try:
            Gx.load_state_dict(torch.load('./genx'))
            Dx.load_state_dict(torch.load('./fcnx'))
            Gy.load_state_dict(torch.load('./geny'))
            Dy.load_state_dict(torch.load('./fcny'))
            print('net loaded')
        except Exception as e:
            print(e)

    dataset = 'ukiyoe2photo'
    # A 562
    image_path_A = './datasets/'+dataset+'/trainA/*.jpg'
    image_path_B = './datasets/'+dataset+'/trainB/*.jpg'

    plt.ion()

    train_image_paths_A = glob.glob(image_path_A)
    train_image_paths_B = glob.glob(image_path_B)
    print(len(train_image_paths_A), len(train_image_paths_B))

    b_size = 8

    train_dataset_A = CustomDataset(
        train_image_paths_A, train=True)
    train_loader_A = torch.utils.data.DataLoader(
        train_dataset_A, batch_size=b_size, shuffle=True, num_workers=4, pin_memory=False, drop_last=True)

    train_dataset_B = CustomDataset(
        train_image_paths_B, True, 562, train=True)
    train_loader_B = torch.utils.data.DataLoader(
        train_dataset_B, batch_size=b_size, shuffle=True, num_workers=4, pin_memory=False, drop_last=True)

    Gx.train()
    Dx.train()

    Gy.train()
    Dy.train()

    criterion = nn.BCEWithLogitsLoss().to(device)
    # criterion2 = nn.SmoothL1Loss().to(device)
    criterion2 = nn.L1Loss().to(device)

    g_lr = 2e-4
    d_lr = 2e-4
    optimizer_x = optim.Adam(Gx.parameters(), lr=g_lr, betas=(0.5, 0.999))
    optimizer_x_d = optim.Adam(Dx.parameters(), lr=d_lr, betas=(0.5, 0.999))

    optimizer_y = optim.Adam(Gy.parameters(), lr=g_lr, betas=(0.5, 0.999))
    optimizer_y_d = optim.Adam(Dy.parameters(), lr=d_lr, betas=(0.5, 0.999))

    # cp = cropper().to(device)

    _zero = torch.from_numpy(
        np.zeros((b_size, 1))).float().to(device)
    _zero.requires_grad = False

    _one = torch.from_numpy(
        np.ones((b_size, 1))).float().to(device)
    _one.requires_grad = False

    for epoch in trange(100, desc='epoch'):
        # loop = tqdm(zip(train_loader_A, train_loader_B), desc='iteration')
        loop = zip(tqdm(train_loader_A, desc='iteration'),
                   train_loader_B)
        batch_idx = 0
        for data_A, data_B in loop:
            batch_idx += 1
            zero = _zero
            one = _one
            _data_A = data_A.to(device)
            _data_B = data_B.to(device)

            # Dy loss (A -> B)
            gen = Gy(_data_A)

            optimizer_y_d.zero_grad()

            output2_p = Dy(_data_B.detach())
            output_p = Dy(gen.detach())

            errD = (criterion(output2_p-torch.mean(output_p), one.detach()) +
                    criterion(output_p-torch.mean(output2_p), zero.detach()))/2
            errD.backward()
            optimizer_y_d.step()

            # Dx loss (B -> A)
            gen = Gx(_data_B)

            optimizer_x_d.zero_grad()

            output2_p = Dx(_data_A.detach())
            output_p = Dx(gen.detach())

            errD = (criterion(output2_p-torch.mean(output_p), one.detach()) +
                    criterion(output_p-torch.mean(output2_p), zero.detach()))/2
            errD.backward()
            optimizer_x_d.step()

            # Gy loss (A -> B)
            optimizer_y.zero_grad()
            gen = Gy(_data_A)
            output_p = Dy(gen)
            output2_p = Dy(_data_B.detach())
            g_loss = (criterion(output2_p-torch.mean(output_p), zero.detach()) +
                      criterion(output_p-torch.mean(output2_p), one.detach()))/2

            # Gy cycle loss (B -> A -> B)
            fA = Gx(_data_B)
            gen = Gy(fA.detach())
            c_loss = criterion2(gen, _data_B)

            errG = g_loss + c_loss
            errG.backward()
            optimizer_y.step()

            if batch_idx % 10 == 0:

                fig = plt.figure(1)
                fig.clf()
                plt.imshow((np.transpose(
                    _data_B.detach().cpu().numpy()[0], (1, 2, 0))+1)/2)
                fig.canvas.draw()
                fig.canvas.flush_events()

                fig = plt.figure(2)
                fig.clf()
                plt.imshow((np.transpose(
                    fA.detach().cpu().numpy()[0], (1, 2, 0))+1)/2)
                fig.canvas.draw()
                fig.canvas.flush_events()

                fig = plt.figure(3)
                fig.clf()
                plt.imshow((np.transpose(
                    gen.detach().cpu().numpy()[0], (1, 2, 0))+1)/2)
                fig.canvas.draw()
                fig.canvas.flush_events()

            # Gx loss (B -> A)
            optimizer_x.zero_grad()
            gen = Gx(_data_B)
            output_p = Dx(gen)
            output2_p = Dx(_data_A.detach())
            g_loss = (criterion(output2_p-torch.mean(output_p), zero.detach()) +
                      criterion(output_p-torch.mean(output2_p), one.detach()))/2

            # Gx cycle loss (A -> B -> A)
            fB = Gy(_data_A)
            gen = Gx(fB.detach())
            c_loss = criterion2(gen, _data_A)

            errG = g_loss + c_loss
            errG.backward()
            optimizer_x.step()

        torch.save(Gx.state_dict(), './genx')
        torch.save(Dx.state_dict(), './fcnx')
        torch.save(Gy.state_dict(), './geny')
        torch.save(Dy.state_dict(), './fcny')
    print('\nFinished Training')


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    random.seed()
    np.random.seed()
    torch.multiprocessing.freeze_support()
    run()
