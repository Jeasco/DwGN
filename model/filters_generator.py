import torch.nn as nn
import torch.nn.functional as F
import torch

class GLGF(nn.Module):
    def __init__(self, inp, oup, fgroup):
        super().__init__()

        self.wn_fc1_l = nn.Conv2d(inp, inp, 1, 1, 0, groups=fgroup, bias=False)
        self.wn_fc1_g1 = nn.Conv2d(inp, fgroup, 1, 1, 0, groups=fgroup, bias=False)
        self.wn_fc1_g2 = nn.Conv2d(fgroup, inp, 1, 1, 0, groups=1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.wn_fc2_l = nn.Conv2d(inp, oup, 1, 1, 0, groups=inp, bias=False)
        self.wn_fc2_g = nn.Conv2d(inp, 1, 1, 1, 0, groups=1, bias=False)

    def forward(self, x):
        x = self.wn_fc1_l(x) + self.wn_fc1_g2(self.wn_fc1_g1(x))
        x = self.sigmoid(x)
        x = self.wn_fc2_l(x) + self.wn_fc2_g(x)

        return x

class WA_1Conv(nn.Module):
    r"""weight adaptive 1 * 1 convolution.
    This layer has inp inputs, inp groups and oup*inp*ksize*ksize outputs.
    """

    def __init__(self, inp, oup, fgroup, ksize=1, stride=1):
        super().__init__()

        self.M = 2
        self.G = 2

        self.pad = ksize // 2
        self.inp = inp
        self.oup = oup
        self.ksize = ksize
        self.stride = stride

        self.glgf = GLGF(inp, oup * inp * ksize * ksize, fgroup)

        self.conv1 = nn.Conv2d(1, 1, 3, 1, 1, groups=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv2d(1, 1, 3, 1, 1, groups=1, bias=False)

    def forward(self, x):
        x_gap = F.adaptive_avg_pool2d(x,(1,1))
        x_w = self.glgf(x_gap)

        if x.shape[0] == 1:  # case of batch size = 1
            x_w = x_w.reshape(self.oup, self.inp, self.ksize, self.ksize)
            x = F.conv2d(x, weight=x_w, stride=self.stride, padding=self.pad)
            return x

        x = x.reshape(1, -1, x.shape[2], x.shape[3])
        x_w = x_w.reshape(-1, self.inp, self.ksize, self.ksize)

        x = F.conv2d(x, weight=x_w, stride=self.stride, padding=self.pad, groups=x_gap.shape[0])
        x = x.reshape(-1, self.oup, x.shape[2], x.shape[3])

        c_gap = x.mean(dim=-3,keepdim=True)
        c = self.conv1(c_gap)
        c = self.sigmoid(c)
        c = self.conv2(c)

        return x * c


class WA_3GConv(nn.Module):
    r""" Here we show a grouping manner when we apply WeightNet to a depthwise convolution.
    The grouped fc layer directly generates the convolutional kernel, has fewer parameters while achieving comparable results.
    This layer has inp inputs, cgroup groups and inp*oup//cgroup*ksize*ksize outputs.
    """

    def __init__(self, inp, oup, fgroup, cgroup, ksize=3, stride=1):
        super().__init__()

        self.M = 2
        self.G = 2

        self.pad = ksize // 2
        self.inp = inp
        self.oup = oup
        self.cgroup = cgroup
        self.ksize = ksize
        self.stride = stride

        self.glgf = GLGF(inp, oup * inp * ksize * ksize // cgroup, fgroup)
        self.glgf_c = GLGF(inp, oup, fgroup)

    def forward(self, x):
        x_gap = F.adaptive_avg_pool2d(x, (1, 1))
        x_w = self.glgf(x_gap)

        # x = x.reshape(1, -1, x.shape[2], x.shape[3])
        # x_w = x_w.reshape(-1, 1, self.ksize, self.ksize)
        # x = F.conv2d(x, weight=x_w, stride=self.stride, padding=self.pad, groups=x_w.shape[0])
        # x = x.reshape(-1, self.oup, x.shape[2], x.shape[3])
        # if x.shape[0] == 1:  # case of batch size = 1
        #     x_w = x_w.reshape(-1, self.inp // self.cgroup, self.ksize, self.ksize)
        #     x = F.conv2d(x, weight=x_w, stride=self.stride, padding=self.pad)
        #     return x

        x = x.reshape(1, -1, x.shape[2], x.shape[3])
        x_w = x_w.reshape(-1, self.inp // self.cgroup, self.ksize, self.ksize)

        x = F.conv2d(x, weight=x_w, stride=self.stride, padding=self.pad, groups=x_gap.shape[0]*self.cgroup)
        x = x.reshape(-1, self.oup, x.shape[2], x.shape[3])

        c = self.glgf_c(x_gap)

        return x * c



if __name__ == '__main__':
    kpn = WA_3GConv(16,16,4,4).cuda()
    a = torch.randn(2, 16, 64, 64).cuda()
    c = torch.randn(2, 16, 1, 1)
    b = kpn(a)
    print(b.shape)
    # filters = torch.randn(2,16, 16, 3, 3)
    # inputs = torch.randn(3, 16, 64, 64)
    # c = F.conv2d(inputs, filters, padding=1)
    # print(c.shape)