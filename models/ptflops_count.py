import argparse

import torch
from ptflops import get_model_complexity_info

from model_defs import NetFinal


def build_model(model_name: str):
    name = model_name.lower()
    if name in ("cifar10", "netbt2_cifar10"):
        return NetFinal(n_class=10), "NetFinal-CIFAR10"
    if name in ("cifar100", "netbt2_cifar100"):
        return NetFinal(n_class=100), "NetFinal-CIFAR100"
    raise ValueError(
        "Unsupported model. Use one of: cifar10, cifar100"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Compute MACs/FLOPs/params with ptflops")
    parser.add_argument("--model", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--print_per_layer", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model, model_label = build_model(args.model)

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device_label = "cuda" if use_cuda else "cpu"

    with torch.cuda.device(0) if use_cuda else torch.no_grad():
        macs, params = get_model_complexity_info(
            model,
            (3, args.image_size, args.image_size),
            as_strings=True,
            print_per_layer_stat=args.print_per_layer,
            verbose=args.print_per_layer,
            flops_units="GMac",
        )

    # 1 MAC ~ 2 FLOPs for common conv/linear operations.
    mac_value, mac_unit = macs.split()
    flops = f"{float(mac_value) * 2:.4f} GFlops" if mac_unit.startswith("G") else f"{float(mac_value) * 2:.4f}"

    print("\n=== PTFlops Summary ===")
    print(f"Model                : {model_label}")
    print(f"Input size           : (3, {args.image_size}, {args.image_size})")
    print(f"Device used          : {device_label}")
    print(f"Computational MACs   : {macs}")
    print(f"Estimated FLOPs      : {flops}")
    print(f"Number of parameters : {params}")
    print(
        "Model parameter count: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )


if __name__ == "__main__":
    main()
