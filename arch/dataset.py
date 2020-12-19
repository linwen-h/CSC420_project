import os
import pathlib
import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random

IMG_EXTENSIONS = set(['.jpg', '.jpeg', '.png', '.ppm', '.bmp', 'tiff'])
def is_image(path):
    return path.suffix.lower() in IMG_EXTENSIONS


class LowResGroundTruthDataset(Dataset):
    """Training Dataset for use when training an SR model."""

    def __init__(self, lr_dir, gt_dir, memcache=False, transform=None,
                 strict_filename_intersection=True):
        super().__init__()
        self._DataLoader__initialized = False
        self.lr_dir = pathlib.Path(lr_dir)
        self.gt_dir = pathlib.Path(gt_dir)
        self.memcache = memcache
        self.transform = transform

        # Attempt filename matching.
        self.lr_image_filepaths = [
            f for f in self.lr_dir.glob('*') if is_image(f)]
        self.gt_image_filepaths = [
            f for f in self.gt_dir.glob('*') if is_image(f)]

        lr_image_filenames = set(
            map(os.path.basename, self.lr_image_filepaths))
        gt_image_filenames = set(
            map(os.path.basename, self.gt_image_filepaths))
        intersect_filenames = lr_image_filenames.intersection(
            gt_image_filenames)
        if strict_filename_intersection:
            mismatched_filenames = (lr_image_filenames.union(
                gt_image_filenames)).difference(intersect_filenames)
            if len(mismatched_filenames) > 0:
                raise ValueError(
                    f"Mismatched filenames in lr_dir and gt_dir: {str(mismatched_filenames)}")

        self.image_filenames = list(sorted(intersect_filenames))
        self.image_lr_gt_pairs = []

        # Load the images if we want to cache them in memory.
        if self.memcache:
            for i, img_filename in enumerate(self.image_filenames):
                # Images with Shape: (C, H, W)
                img_lr = Image.open(os.path.join(
                    self.lr_dir, img_filename)).convert("RGB")
                img_gt = Image.open(os.path.join(
                    self.gt_dir, img_filename)).convert("RGB")
                self.image_lr_gt_pairs.append((img_lr, img_gt))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, i):
        img_filename = self.image_filenames[i]
        if self.memcache:
            img_lr, img_gt = self.image_lr_gt_pairs[i]
        else:
            # Images with Shape: (C, H, W)
            img_lr = Image.open(os.path.join(
                self.lr_dir, img_filename)).convert("RGB")
            img_gt = Image.open(os.path.join(
                self.gt_dir, img_filename)).convert("RGB")

        # Apply Data Augmentation
        if self.transform is not None:
            # Use set seed to make sure both LR and GT images get same transforms
            seed = random.randint(0, 1e7)
            torch.manual_seed(seed)
            img_lr = self.transform(img_lr)
            torch.manual_seed(seed)
            img_gt = self.transform(img_gt)
        else:
            img_lr = torchvision.transforms.ToTensor()(img_lr)
            img_gt = torchvision.transforms.ToTensor()(img_gt)

        # Apply Normalization from [0, 1] -> [-1, 1]
        img_lr = (img_lr * 2) - 1.0
        img_gt = (img_gt * 2) - 1.0

        return {
            'img_filename': img_filename,
            'img_lr': img_lr,
            'img_gt': img_gt
        }


class LowResDataSet(Dataset):
    """Dataset for use when performing inference."""
    def __init__(self, lr_dir, memcache=False):
        super().__init__()
        self.lr_dir = pathlib.Path(lr_dir)
        self.memcache = memcache

        self.lr_image_filepaths = [f for f in lr_dir.glob('*') if is_image(f)]
        self.image_filenames = sorted(list(map(os.path.basename, self.lr_image_filepaths)))
        self.image_lr = []

        # Load the images if we want to cache them in memory.
        if self.memcache:
            for i, img_filename in enumerate(self.image_filenames):
                # Images with Shape: (C, H, W)
                img_lr = Image.open(os.path.join(
                    self.lr_dir, img_filename)).convert("RGB")
                self.image_lr.append(img_lr)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, i):
        img_filename = self.image_filenames[i]
        if self.memcache:
            img_lr = self.image_lr[i]
        else:
            img_lr = Image.open(os.path.join(
                self.lr_dir, img_filename)).convert("RGB")
        
        img_lr = torchvision.transforms.ToTensor()(img_lr)
        # Apply Normalization from [0, 1] -> [-1, 1]
        img_lr = (img_lr * 2) - 1.0

        return {
            'img_filename': img_filename,
            'img_lr': img_lr
        }
