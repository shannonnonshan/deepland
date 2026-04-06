import torch
import torch.nn as nn
import torch.nn.functional as F

from SE_attention import SE


class PDPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, s):
        super(PDPBlock, self).__init__()
        self.pw1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.dw1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=s, padding=1, groups=in_channels, dilation=1)
        self.dw2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=s, padding=2, groups=in_channels, dilation=2)
        self.pw2 = nn.Conv2d(in_channels=2*in_channels, out_channels=out_channels, kernel_size=1)
        self.SE = SE(out_channels, 16)
        self.s = s
        self.PwR = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=s)
    def forward(self, x):
        Pw1 = self.pw1(x)
        Dw1 = F.relu(self.dw1(Pw1))
        Dw2 = F.relu(self.dw2(Pw1))
        Dw = torch.cat([Dw1, Dw2], dim=1)
        DSDC = self.shuffle(Dw)
        DSDC = self.pw2(DSDC)
        PDP = self.SE(DSDC)
        if self.s == 1 and x.size() == PDP.size():
            FRPDP = x + PDP
        else:
            PwR = F.relu(self.PwR(x))
            FRPDP = PwR + PDP
        return F.relu(FRPDP)
    def shuffle(self, x):
      num_group = 2
      h, num_channel, height, width = x.data.size()
      group_channels = num_channel // num_group

      x = x.reshape(h, num_group, group_channels, height, width)
      x = x.permute(0, 2, 1, 3, 4)
      x = x.reshape(h, num_channel, height, width)
      return x

class NetMid(nn.Module):
    def __init__(self, image_size, n_class=10):
        super(NetMid, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.block1 = PDPBlock(in_channels=32, out_channels=64, s=1)
        self.block2 = PDPBlock(in_channels=64, out_channels=64, s=1)
        self.block3 = PDPBlock(in_channels=64, out_channels=128, s=1 if image_size == 32 else 2)
        self.block5 = PDPBlock(in_channels=128, out_channels=128, s=1)
        self.block6 = PDPBlock(in_channels=128, out_channels=256, s=2)
        self.block7 = PDPBlock(in_channels=256, out_channels=256, s=1)
        self.block8 = PDPBlock(in_channels=256, out_channels=256, s=2)
        self.block9 = PDPBlock(in_channels=256, out_channels=512, s=1)
        self.block10 = PDPBlock(in_channels=512, out_channels=512, s=2)
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
        self.block1 = PDPBlock(in_channels=32, out_channels=64, s=1)
        self.block2 = PDPBlock(in_channels=64, out_channels=64, s=1)
        self.block3 = PDPBlock(in_channels=64, out_channels=128, s=2)
        self.block5 = PDPBlock(in_channels=128, out_channels=128, s=1)
        self.block6 = PDPBlock(in_channels=128, out_channels=256, s=2)
        self.block7 = PDPBlock(in_channels=256, out_channels=256, s=1)
        self.block8 = PDPBlock(in_channels=256, out_channels=256, s=1)
        self.block9 = PDPBlock(in_channels=256, out_channels=512, s=2)
        self.block10 = PDPBlock(in_channels=512, out_channels=512, s=1)
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
