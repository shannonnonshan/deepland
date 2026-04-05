import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model_defs import NetFinal
from train_config import VERIFY_CONFIG
from train_utils import evaluate

CHECKPOINT = "outputs_final/cifar10/best_checkpoint.pth"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    test_ds = datasets.CIFAR10(root="./data", train=False, transform=test_tf, download=True)
    test_loader = DataLoader(
        test_ds,
        batch_size=VERIFY_CONFIG["batch_size"],
        shuffle=False,
        num_workers=VERIFY_CONFIG["num_workers"],
    )

    model = NetFinal(n_class=10).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Checkpoint: {CHECKPOINT}")
    print("Dataset: CIFAR-10")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
