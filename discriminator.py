import numpy as np

import torch
import torch.nn as nn
from cbam import cbam


class Discriminator(nn.Module):
    def __init__(self, output_dim=1):
        super(Discriminator, self).__init__()
        self.output_dim = output_dim
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 32, 4, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, 4, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, self.output_dim, 3),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        '''
        self.fc = nn.Sequential(
            # nn.Linear(128, 2),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
            # nn.ReLU(inplace=True),
            # nn.Linear(128, 2),
        )
        '''

    def forward(self, x):
        f = self.conv(x)
        x = self.gap(f)
        x = x.view(-1, self.output_dim)
        # x = self.fc(x)
        # x = torch.sigmoid(x)
        return x

if __name__ == "__main__":
    net = Discriminator()
    d = net.state_dict()
    print(d)
    for name, param in d.items():
        print(name, param)

    # z = torch.from_numpy(np.zeros((1, 3, 538, 438))).float()
    # output = net(z)
    # print(output.shape)
