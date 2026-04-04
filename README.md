# deepland

Final project for Deep Learning, 2026.

## Project structure

- Midterm model M1: models/mid-model.py
- Final model M2: models/final-model.py
- Dataset D loader: models/datasets.py
- Verify checkpoint (CIFAR-10): models/testNet_CIFAR10.py
- Verify checkpoint (CIFAR-100): models/testNet_CIFAR100.py

## Midterm test (M1 on Dataset D)

Requirements covered:

- Dataset D with at least 5 classes
- Train M1 about 100 epochs on image size 32x32 and 224x224
- Log per-epoch metrics: train/test accuracy and train/test loss
- Save charts for accuracy/loss
- Save best checkpoints and ranked results

Dataset layout supported:

- datasets/train/<class_name>
- datasets/var/<class_name>

Run:

python models/mid-model.py --data_root datasets --epochs 100 --batch_size 64 --output_root outputs_midterm

Output files:

- outputs_midterm/img_32/epoch_log.txt
- outputs_midterm/img_32/history.csv
- outputs_midterm/img_32/curves.png
- outputs_midterm/img_32/best_checkpoint.pth
- outputs_midterm/img_224/epoch_log.txt
- outputs_midterm/img_224/history.csv
- outputs_midterm/img_224/curves.png
- outputs_midterm/img_224/best_checkpoint.pth

## Final test (M2 on CIFAR-10 and CIFAR-100)

Requirements covered:

- Design CNN model M2 (reused and adapted from M1)
- Train on CIFAR-10 and CIFAR-100 about 200 epochs
- Log per-epoch metrics: train/test accuracy and train/test loss
- Save charts for accuracy/loss
- Save best checkpoints and ranked results

Run:

python models/final-model.py --epochs 200 --batch_size 128 --output_root outputs_final

Output files:

- outputs_final/cifar10/epoch_log.txt
- outputs_final/cifar10/history.csv
- outputs_final/cifar10/curves.png
- outputs_final/cifar10/best_checkpoint.pth
- outputs_final/cifar100/epoch_log.txt
- outputs_final/cifar100/history.csv
- outputs_final/cifar100/curves.png
- outputs_final/cifar100/best_checkpoint.pth

## Verify best checkpoints

Verify CIFAR-10 best checkpoint:

python models/testNet_CIFAR10.py --checkpoint outputs_final/cifar10/best_checkpoint.pth

Verify CIFAR-100 best checkpoint:

python models/testNet_CIFAR100.py --checkpoint outputs_final/cifar100/best_checkpoint.pth

## Report notes

- This repository uses README as the report document.
- Ranked results are appended into README automatically when training scripts complete.

## Final Report (M2 on CIFAR-10 and CIFAR-100)

- Training logs per epoch are saved in outputs_final/cifar10 and outputs_final/cifar100
- Charts are saved as curves.png in each folder
- Best checkpoint files are saved as best_checkpoint.pth

Ranked results:
1. dataset=CIFAR10, best_test_acc=40.67%, best_epoch=1
2. dataset=CIFAR100, best_test_acc=7.30%, best_epoch=1
