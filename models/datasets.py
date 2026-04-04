import os
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def set_seed(seed: int = 42) -> None:
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def _build_transforms(image_size: int):
	train_tf = transforms.Compose(
		[
			transforms.Resize((image_size, image_size)),
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.RandomRotation(10),
			transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)
	test_tf = transforms.Compose(
		[
			transforms.Resize((image_size, image_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)
	return train_tf, test_tf


def _is_valid_image_file(path: str) -> bool:
	try:
		with Image.open(path) as img:
			img.load()
		return True
	except Exception:
		return False


class SafeImageFolder(datasets.ImageFolder):
	def __getitem__(self, index):
		# Skip corrupted/unreadable files instead of crashing DataLoader workers.
		for _ in range(10):
			try:
				return super().__getitem__(index)
			except Exception:
				index = (index + 1) % len(self.samples)
		raise RuntimeError("Too many corrupted images encountered consecutively.")


def _find_dataset_layout(data_root: str):
	root = Path(data_root)
	train_dir = root / "train"
	test_dir = root / "test"
	val_dir = root / "val"
	var_dir = root / "var"

	if train_dir.exists() and (test_dir.exists() or val_dir.exists() or var_dir.exists()):
		if test_dir.exists():
			eval_dir = test_dir
		elif val_dir.exists():
			eval_dir = val_dir
		else:
			eval_dir = var_dir
		return "separate", str(train_dir), str(eval_dir)
	return "single", str(root), None


def get_dataloaders(
	data_root: str,
	image_size: int,
	batch_size: int = 64,
	num_workers: int = 2,
	test_split: float = 0.2,
	seed: int = 42,
):
	"""
	Build train/test dataloaders from an ImageFolder dataset.

	Supported directory layouts:
	1) data_root/train/<class_name>, data_root/test/<class_name>
	2) data_root/<class_name>  (auto split by test_split)
	"""
	set_seed(seed)
	train_tf, test_tf = _build_transforms(image_size)
	layout, train_path, eval_path = _find_dataset_layout(data_root)

	if layout == "separate":
		train_dataset = SafeImageFolder(train_path, transform=train_tf, is_valid_file=_is_valid_image_file)
		test_dataset = SafeImageFolder(eval_path, transform=test_tf, is_valid_file=_is_valid_image_file)
		class_names = train_dataset.classes
	else:
		full_for_split = SafeImageFolder(train_path, transform=None, is_valid_file=_is_valid_image_file)
		n_total = len(full_for_split)
		n_test = int(n_total * test_split)
		n_train = n_total - n_test
		generator = torch.Generator().manual_seed(seed)
		train_subset, test_subset = random_split(full_for_split, [n_train, n_test], generator=generator)

		# Create two independent datasets to use different transforms.
		train_dataset = SafeImageFolder(train_path, transform=train_tf, is_valid_file=_is_valid_image_file)
		test_dataset = SafeImageFolder(train_path, transform=test_tf, is_valid_file=_is_valid_image_file)
		train_dataset = torch.utils.data.Subset(train_dataset, train_subset.indices)
		test_dataset = torch.utils.data.Subset(test_dataset, test_subset.indices)
		class_names = full_for_split.classes

	n_classes = len(class_names)
	if n_classes < 5:
		raise ValueError(
			f"Dataset has {n_classes} classes. Requirement needs at least 5 classes."
		)

	pin_memory = torch.cuda.is_available()
	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=pin_memory,
	)
	test_loader = DataLoader(
		test_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=pin_memory,
	)

	return train_loader, test_loader, class_names


def infer_num_classes(data_root: str) -> int:
	layout, train_path, _ = _find_dataset_layout(data_root)
	if layout == "separate":
		ds = SafeImageFolder(train_path, is_valid_file=_is_valid_image_file)
	else:
		ds = SafeImageFolder(train_path, is_valid_file=_is_valid_image_file)
	return len(ds.classes)


if __name__ == "__main__":
	default_root = os.environ.get("DATA_ROOT", "./data")
	for size in (32, 224):
		tr, te, classes = get_dataloaders(default_root, image_size=size, batch_size=16)
		print(f"Image size {size}: train batches={len(tr)}, test batches={len(te)}, classes={len(classes)}")
