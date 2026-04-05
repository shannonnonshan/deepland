import torch
import torch.nn as nn
import torch.nn.functional as F

from SE_attention import SE


class PDPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(PDPBlock, self).__init__()
        self.pw1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.dw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
        )
        self.pw2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.se = SE(out_channels, 16)
        self.stride = stride
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x_pw = self.pw1(x)
        x_dw = F.relu(self.dw(x_pw))
        x_pw2 = self.pw2(x_dw)
        x_attn = self.se(x_pw2)

        if self.stride == 1 and x.size() == x_attn.size():
            x_out = x + x_attn
        else:
            x_out = F.relu(self.proj(x)) + x_attn
        return F.relu(x_out)


class NetMid(nn.Module):
    def __init__(self, image_size, n_class=10):
        super(NetMid, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.block1 = PDPBlock(in_channels=32, out_channels=64, stride=1)
        self.block2 = PDPBlock(in_channels=64, out_channels=64, stride=1)
        self.block3 = PDPBlock(in_channels=64, out_channels=128, stride=1 if image_size == 32 else 2)
        self.block5 = PDPBlock(in_channels=128, out_channels=128, stride=1)
        self.block6 = PDPBlock(in_channels=128, out_channels=256, stride=2)
        self.block7 = PDPBlock(in_channels=256, out_channels=256, stride=1)
        self.block8 = PDPBlock(in_channels=256, out_channels=256, stride=2)
        self.block9 = PDPBlock(in_channels=256, out_channels=512, stride=1)
        self.block10 = PDPBlock(in_channels=512, out_channels=512, stride=2)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(1024, n_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


class NetFinal(nn.Module):
    def __init__(self, n_class=10):
        super(NetFinal, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.block1 = PDPBlock(in_channels=32, out_channels=64, stride=1)
        self.block2 = PDPBlock(in_channels=64, out_channels=64, stride=1)
        self.block3 = PDPBlock(in_channels=64, out_channels=128, stride=2)
        self.block5 = PDPBlock(in_channels=128, out_channels=128, stride=1)
        self.block6 = PDPBlock(in_channels=128, out_channels=256, stride=2)
        self.block7 = PDPBlock(in_channels=256, out_channels=256, stride=1)
        self.block8 = PDPBlock(in_channels=256, out_channels=256, stride=1)
        self.block9 = PDPBlock(in_channels=256, out_channels=512, stride=2)
        self.block10 = PDPBlock(in_channels=512, out_channels=512, stride=1)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.dropout = nn.Dropout(p=0.25)
        self.fc = nn.Linear(1024, n_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)
