import json
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model_defs import NetFinal
from train_config import FINAL_CONFIG
from train_utils import run_one_epoch, save_history_csv, save_plots


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def train_cifar(dataset_name, epochs, batch_size, learning_rate, weight_decay, num_workers, seed, output_root):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, n_class = get_cifar_loaders(dataset_name, batch_size, num_workers)

    model = NetFinal(n_class=n_class).to(device)
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
    save_plots(history, curve_png, title_prefix=dataset_name)

    summary_dict = {
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
        json.dump(summary_dict, f, indent=2)

    return summary_dict


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
    cfg = FINAL_CONFIG
    os.makedirs(cfg["output_root"], exist_ok=True)

    summary_10 = train_cifar(
        dataset_name="CIFAR10",
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        num_workers=cfg["num_workers"],
        seed=cfg["seed"],
        output_root=cfg["output_root"],
    )

    summary_100 = train_cifar(
        dataset_name="CIFAR100",
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        num_workers=cfg["num_workers"],
        seed=cfg["seed"],
        output_root=cfg["output_root"],
    )

    update_readme_final("README.md", summary_10, summary_100)

    print("\n=== FINAL BEST RESULTS (M2) ===")
    print(f"CIFAR10  -> best test acc: {summary_10['best_test_acc']:.2f}% at epoch {summary_10['best_epoch']}")
    print(f"CIFAR100 -> best test acc: {summary_100['best_test_acc']:.2f}% at epoch {summary_100['best_epoch']}")


if __name__ == "__main__":
    main()
