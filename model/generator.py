import torch.nn as nn
import torch
import numpy as np
import functools
from torch.nn import init
import torch.nn.functional as F



#G_b
class G_b(nn.Module):
    def __init__(self, ndf=32, input_dim=3, nGMM=10):
        super(G_b, self).__init__()
        self.nGMM = nGMM

        self.input_pro = nn.Conv2d(input_dim, ndf, kernel_size=4, stride=1, padding=2, bias=False)
        # Layer1 128
        self.layer1 = nn.Sequential(
            nn.Conv2d(ndf, ndf, kernel_size=4, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True)

        )

        # Layer2 64
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

        )

        # Layer3 32
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

        )

        # Layer4 16
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 16, kernel_size=4, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.PixelShuffle(2)
        )

        # Layer5 32
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.PixelShuffle(2)

        )

        # Layer6 64
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.PixelShuffle(2)

        )

        # Layer7 128
        self.layer6 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, self.nGMM, 1, 1, 0, bias=False),
            nn.Sigmoid()

        )

        self.cGMM = torch.nn.Parameter(torch.FloatTensor(self.nGMM*2), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self,x):
        x = self.input_pro(x)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out = F.gumbel_softmax(out7, tau=0.1, hard=True, dim=1)
        map_b = torch.zeros_like(x)
        for i in range(self.nGMM):
            map_b += out[:,i,:,:] * (torch.randn(x.shape[0], x.shape[1], x.shape[2], x.shape[3]) * self.cGMM[2*i] + self.cGMM[2*i+1])

        return map_b

#G_d
class G_d(nn.Module):
    def __init__(self, ndf=32, input_dim=6, nGMM=10):
        super(G_d, self).__init__()
        self.nGMM = nGMM

        self.input_pro = nn.Conv2d(input_dim, ndf, kernel_size=4, stride=1, padding=2, bias=False)
        # Layer1 128
        self.layer1 = nn.Sequential(
            nn.Conv2d(ndf, ndf, kernel_size=4, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True)

        )

        # Layer2 64
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

        )

        # Layer3 32
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

        )

        # Layer4 16
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 16, kernel_size=4, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.PixelShuffle(2)
        )

        # Layer5 32
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.PixelShuffle(2)

        )

        # Layer6 64
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.PixelShuffle(2)

        )

        # Layer7 128
        self.layer6 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, self.nGMM, 1, 1, 0, bias=False),
            nn.Sigmoid()

        )

        self.cGMM = torch.nn.Parameter(torch.FloatTensor(self.nGMM * 2), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = self.input_pro(x)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out = F.gumbel_softmax(out7, tau=0.1, hard=True, dim=1)
        map_d = torch.zeros_like(x)
        for i in range(self.nGMM):
            map_d += out[:, i, :, :] * (
                        torch.randn(x.shape[0], x.shape[1], x.shape[2], x.shape[3]) * self.cGMM[2 * i] + self.cGMM[
                    2 * i + 1])

        return map_d
