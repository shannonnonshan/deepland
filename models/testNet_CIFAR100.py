import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from SE_attention import SE


class modulePDP(nn.Module):
    def __init__(self, in_channels, out_channels, s):
        super(modulePDP, self).__init__()
        self.pw1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.dw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=s,
            padding=1,
            groups=in_channels,
        )
        self.pw2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.SE = SE(out_channels, 16)
        self.s = s
        self.PwR = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=s)

    def forward(self, x):
        Pw1 = self.pw1(x)
        Dw = F.relu(self.dw(Pw1))
        Pw2 = self.pw2(Dw)
        PDP = self.SE(Pw2)
        if self.s == 1 and x.size() == PDP.size():
            FRPDP = x + PDP
        else:
            PwR = F.relu(self.PwR(x))
            FRPDP = PwR + PDP
        return F.relu(FRPDP)


class NetBT2(nn.Module):
    def __init__(self, n_class=100):
        super(NetBT2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.modulesPDP1 = modulePDP(in_channels=32, out_channels=64, s=1)
        self.modulesPDP2 = modulePDP(in_channels=64, out_channels=64, s=1)
        self.modulesPDP3 = modulePDP(in_channels=64, out_channels=128, s=2)
        self.modulesPDP5 = modulePDP(in_channels=128, out_channels=128, s=1)
        self.modulesPDP6 = modulePDP(in_channels=128, out_channels=256, s=2)
        self.modulesPDP7 = modulePDP(in_channels=256, out_channels=256, s=1)
        self.modulesPDP8 = modulePDP(in_channels=256, out_channels=256, s=1)
        self.modulesPDP9 = modulePDP(in_channels=256, out_channels=512, s=2)
        self.modulesPDP10 = modulePDP(in_channels=512, out_channels=512, s=1)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(1024, n_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.modulesPDP1(x)
        x = self.modulesPDP2(x)
        x = self.modulesPDP3(x)
        x = self.modulesPDP5(x)
        x = self.modulesPDP6(x)
        x = self.modulesPDP7(x)
        x = self.modulesPDP8(x)
        x = self.modulesPDP9(x)
        x = self.modulesPDP10(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


def evaluate(model, loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total, correct = 0, 0
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
    return running_loss / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description="Verify best checkpoint for CIFAR-100")
    parser.add_argument("--checkpoint", type=str, default="outputs_final/cifar100/best_checkpoint.pth")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    test_ds = datasets.CIFAR100(root="./data", train=False, transform=test_tf, download=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = NetBT2(n_class=100).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Checkpoint: {args.checkpoint}")
    print("Dataset: CIFAR-100")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
