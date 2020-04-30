import numpy as np
import torch
import torch.nn as nn

from cbam import cbam, cbam_channel


class Generator_v3(nn.Module):
    def __init__(self, img_size=64, latent_length=64, hidden_length=256):
        super(Generator_v3, self).__init__()
        self.init_size = img_size // 4
        self.latent = latent_length
        self.encode = nn.Sequential(
            nn.Linear(self.latent, hidden_length, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_length, self.latent*8*2*2, bias=False),
            nn.ReLU(inplace=True),
        )
        self.decode = nn.Sequential(
            nn.Linear(self.latent*8*2*2, hidden_length, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_length, self.latent, bias=False),
        )
        self.conv = self._make_layer(self.latent*8, 4)
        self.out_conv = nn.Sequential(
            nn.Conv2d(self.latent//2, 3, 3,
                      stride=1, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encode(x)
        z = self.decode(x)
        x = x.view(x.shape[0], self.latent*8, 2, 2)
        x = self.conv(x)
        x = self.out_conv(x)
        return x, z

    def _make_layer(self, input_dim, n):
        layers = []
        for l in range(n):
            layers.append(nn.ConvTranspose2d(
                input_dim, input_dim//2, 4, 2, bias=False))

            input_dim = input_dim//2
            layers.append(nn.BatchNorm2d(input_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        return nn.Sequential(*layers)


if __name__ == "__main__":
    pass
