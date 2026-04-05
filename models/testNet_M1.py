import torch

from datasets import get_dataloaders
from model_defs import NetMid
from train_config import MID_CONFIG, VERIFY_CONFIG
from train_utils import evaluate

M1_CHECKPOINT_32 = "outputs_midterm/img_32/best_checkpoint.pth"
M1_CHECKPOINT_224 = "outputs_midterm/img_224/best_checkpoint.pth"


def verify_one(checkpoint_path, image_size, data_root, device):
    _, test_loader, class_names = get_dataloaders(
        data_root=data_root,
        image_size=image_size,
        batch_size=VERIFY_CONFIG["batch_size"],
        num_workers=VERIFY_CONFIG["num_workers"],
        print_summary=False,
    )

    ckpt = torch.load(checkpoint_path, map_location=device)
    n_class = ckpt.get("n_class", len(class_names))

    model = NetMid(image_size=image_size, n_class=n_class).to(device)
    model.load_state_dict(ckpt["model_state"])

    test_loss, test_acc = evaluate(model, test_loader, device)
    print("=== Verify M1 Checkpoint ===")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Image size : {image_size}")
    print(f"Classes    : {n_class}")
    print(f"Test Loss  : {test_loss:.4f}")
    print(f"Test Acc   : {test_acc:.2f}%")
    print()


def main():
    data_root = MID_CONFIG["data_root"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    verify_one(M1_CHECKPOINT_32, 32, data_root, device)
    verify_one(M1_CHECKPOINT_224, 224, data_root, device)


if __name__ == "__main__":
    main()
