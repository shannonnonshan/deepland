import argparse
import csv
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from SE_attention import SE


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    def __init__(self, image_size=32, n_class=10):
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
        self.dropout = nn.Dropout(p=0.25)
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
        x = self.dropout(x)
        x = self.fc1(x)
        return x


def get_cifar_loaders(dataset_name, batch_size, num_workers):
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    root = "./data"
    if dataset_name.lower() == "cifar10":
        train_ds = datasets.CIFAR10(root=root, train=True, transform=train_tf, download=True)
        test_ds = datasets.CIFAR10(root=root, train=False, transform=test_tf, download=True)
        n_class = 10
    else:
        train_ds = datasets.CIFAR100(root=root, train=True, transform=train_tf, download=True)
        test_ds = datasets.CIFAR100(root=root, train=False, transform=test_tf, download=True)
        n_class = 100

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader, n_class


def run_one_epoch(model, loader, criterion, optimizer, device, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


def save_history_csv(history, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc"])
        for i in range(len(history["train_loss"])):
            writer.writerow(
                [
                    i + 1,
                    history["train_loss"][i],
                    history["train_acc"][i],
                    history["test_loss"][i],
                    history["test_acc"][i],
                ]
            )


def save_plots(history, save_png, title_prefix):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["test_acc"], label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{title_prefix} Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_png, dpi=200)
    plt.close()


def train_cifar(dataset_name, epochs, batch_size, learning_rate, weight_decay, num_workers, seed, output_root):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, n_class = get_cifar_loaders(dataset_name, batch_size, num_workers)

    model = NetBT2(image_size=32, n_class=n_class).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    exp_dir = Path(output_root) / dataset_name.lower()
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = exp_dir / "best_checkpoint.pth"
    csv_path = exp_dir / "history.csv"
    txt_log = exp_dir / "epoch_log.txt"
    curve_png = exp_dir / "curves.png"

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best = {"epoch": 0, "test_acc": 0.0, "test_loss": 99999.0}

    with open(txt_log, "w", encoding="utf-8") as f:
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
            test_loss, test_acc = run_one_epoch(model, test_loader, criterion, optimizer, device, is_train=False)
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc)

            log_line = (
                f"[{dataset_name}] Epoch [{epoch}/{epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
            )
            print(log_line)
            f.write(log_line + "\n")

            if test_acc > best["test_acc"]:
                best = {"epoch": epoch, "test_acc": test_acc, "test_loss": test_loss}
                torch.save(
                    {
                        "dataset": dataset_name,
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "n_class": n_class,
                        "image_size": 32,
                        "best_test_acc": test_acc,
                    },
                    ckpt_path,
                )

    save_history_csv(history, csv_path)
    save_plots(history, curve_png, dataset_name)

    summary = {
        "dataset": dataset_name,
        "epochs": epochs,
        "best_epoch": best["epoch"],
        "best_test_acc": best["test_acc"],
        "best_test_loss": best["test_loss"],
        "checkpoint": str(ckpt_path),
        "history_csv": str(csv_path),
        "curves_png": str(curve_png),
    }
    with open(exp_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def update_readme_final(readme_path, sum10, sum100):
    ranked = sorted([sum10, sum100], key=lambda x: x["best_test_acc"], reverse=True)
    section = [
        "",
        "## Final Report (M2 on CIFAR-10 and CIFAR-100)",
        "",
        "- Training logs per epoch are saved in outputs_final/cifar10 and outputs_final/cifar100",
        "- Charts are saved as curves.png in each folder",
        "- Best checkpoint files are saved as best_checkpoint.pth",
        "",
        "Ranked results:",
    ]
    for i, item in enumerate(ranked, 1):
        section.append(
            f"{i}. dataset={item['dataset']}, best_test_acc={item['best_test_acc']:.2f}%, best_epoch={item['best_epoch']}"
        )

    with open(readme_path, "a", encoding="utf-8") as f:
        f.write("\n".join(section) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_root", type=str, default="outputs_final")
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    summary_10 = train_cifar(
        dataset_name="CIFAR10",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        output_root=args.output_root,
    )

    summary_100 = train_cifar(
        dataset_name="CIFAR100",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        output_root=args.output_root,
    )

    update_readme_final("README.md", summary_10, summary_100)

    print("\n=== FINAL BEST RESULTS (M2) ===")
    print(f"CIFAR10  -> best test acc: {summary_10['best_test_acc']:.2f}% at epoch {summary_10['best_epoch']}")
    print(f"CIFAR100 -> best test acc: {summary_100['best_test_acc']:.2f}% at epoch {summary_100['best_epoch']}")


if __name__ == "__main__":
    main()
