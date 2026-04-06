import json
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from datasets import get_dataloaders
from model_defs import NetMid
from train_config import MID_CONFIG
from train_utils import run_one_epoch, save_history_csv, save_plots


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_single_config(
    data_root,
    image_size,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    num_workers,
    seed,
    output_root,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, class_names = get_dataloaders(
        data_root=data_root,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        print_summary=True,
    )
    n_class = len(class_names)

    model = NetMid(image_size=image_size, n_class=n_class).to(device)
    print(model)
    summary(model, (3, image_size, image_size))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    exp_dir = Path(output_root) / f"img_{image_size}"
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
                f"Epoch [{epoch}/{epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
            )
            print(log_line)
            f.write(log_line + "\n")

            if test_acc > best["test_acc"]:
                best = {"epoch": epoch, "test_acc": test_acc, "test_loss": test_loss}
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "class_names": class_names,
                        "image_size": image_size,
                        "n_class": n_class,
                        "best_test_acc": test_acc,
                    },
                    ckpt_path,
                )

    save_history_csv(history, csv_path)
    save_plots(history, curve_png, title_prefix="")

    result_summary = {
        "image_size": image_size,
        "epochs": epochs,
        "n_class": n_class,
        "best_epoch": best["epoch"],
        "best_test_acc": best["test_acc"],
        "best_test_loss": best["test_loss"],
        "checkpoint": str(ckpt_path),
        "history_csv": str(csv_path),
        "curves_png": str(curve_png),
    }
    with open(exp_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(result_summary, f, indent=2)

    return result_summary


def main():
    cfg = MID_CONFIG
    set_seed(cfg["seed"])
    os.makedirs(cfg["output_root"], exist_ok=True)

    summary = train_single_config(
        data_root=cfg["data_root"],
        image_size=32,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        num_workers=cfg["num_workers"],
        seed=cfg["seed"],
        output_root=cfg["output_root"],
    )

    print("\n=== MIDTERM RESULTS (M1) - 32x32 ===")
    print(f"Best test acc: {summary['best_test_acc']:.2f}% at epoch {summary['best_epoch']}")


if __name__ == "__main__":
    main()
