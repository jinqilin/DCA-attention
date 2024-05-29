import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class RFDCAConv(nn.Module): 
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, reduction =32):
        super().__init__()
        self.kernel_size = kernel_size
        self.generate = nn.Sequential(nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size, padding=kernel_size // 2,
                                                stride=stride, groups=in_channel,
                                                bias=False),
                                      nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
                                      nn.ReLU()
                                      )
        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, stride=kernel_size))

        self.pool_all = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, in_channel // reduction)
        self.conv1 = nn.Conv2d(in_channel, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, in_channel, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channel, kernel_size=1, stride=1, padding=0)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c = x.shape[0:2]
        generate_feature = self.generate(x)
        h, w = generate_feature.shape[2:]
        generate_feature = generate_feature.view(b, c, self.kernel_size ** 2, h, w)

        generate_feature = rearrange(generate_feature, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                                     n2=self.kernel_size)
        x_cha = torch.mean(generate_feature, dim=1, keepdim=True) - generate_feature
        x_c = self.pool_all(x_cha).sigmoid()
        x_h = self.pool_h(generate_feature)
        x_w = self.pool_w(generate_feature).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        h, w = generate_feature.shape[2:]
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        a_out = self.conv(generate_feature*x_c*a_h * a_w)
        return a_out