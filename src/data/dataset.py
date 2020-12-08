import os
import pathlib
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random

from data.util import is_image


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
        self.lr_image_filepaths = [f for f in self.lr_dir.glob('*') if is_image(f)]
        self.gt_image_filepaths = [f for f in self.gt_dir.glob('*') if is_image(f)]

        lr_image_filenames = set(map(os.path.basename, self.lr_image_filepaths))
        gt_image_filenames = set(map(os.path.basename, self.gt_image_filepaths))
        intersect_filenames = lr_image_filenames.intersection(gt_image_filenames)
        if strict_filename_intersection:
            mismatched_filenames = (lr_image_filenames.union(gt_image_filenames)).difference(intersect_filenames)
            if len(mismatched_filenames) > 0:
                raise ValueError(f"Mismatched filenames in lr_dir and gt_dir: {str(mismatched_filenames)}")

        self.image_filenames = list(sorted(intersect_filenames))
        self.image_lr_gt_pairs = []

        # Load the images if we want to cache them in memory.
        if self.memcache:
            for i, img_filename in enumerate(self.image_filenames):
                img_lr = np.array(Image.open(os.path.join(self.lr_dir, img_filename)).convert("RGB")).astype(np.uint8)
                img_gt = np.array(Image.open(os.path.join(self.gt_dir, img_filename)).convert("RGB")).astype(np.uint8)
                img_lr = (img_lr / 127.5) - 1.0
                img_gt = (img_gt / 127.5) - 1.0
                if self.transform is not None:
                    img_lr, img_gt = self.transform((img_lr, img_gt))
                img_lr = img_lr.transpose(2, 0, 1).astype(np.float32)
                img_gt = img_gt.transpose(2, 0, 1).astype(np.float32)
                self.image_lr_gt_pairs.append((img_lr, img_gt))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, i):
        img_filename = self.image_filenames[i]
        if self.memcache:
            img_lr, img_gt = self.image_lr_gt_pairs[i]
        else:
            img_lr = np.array(Image.open(os.path.join(self.lr_dir, img_filename)).convert("RGB")).astype(np.uint8)
            img_gt = np.array(Image.open(os.path.join(self.gt_dir, img_filename)).convert("RGB")).astype(np.uint8)
            img_lr = (img_lr / 127.5) - 1.0
            img_gt = (img_gt / 127.5) - 1.0

            if self.transform is not None:
                img_lr, img_gt = self.transform((img_lr, img_gt))

            img_lr = img_lr.transpose(2, 0, 1).astype(np.float32)
            img_gt = img_gt.transpose(2, 0, 1).astype(np.float32)
        return {
            'img_filename': img_filename,
            'img_lr': img_lr,
            'img_gt': img_gt
        }


class LowResDataSet(Dataset):
    def __init__(self, lr_dir, memcache=False):
        super().__init__()
        self.lr_dir = lr_dir
        self.memcache = memcache

        # Attempt filename matching.
        self.lr_image_filepaths = [f for f in lr_dir.glob('*') if is_image(f)]
        self.image_filenames = sorted(list(map(os.path.basename, self.lr_image_filepaths)))
        self.image_lr = []

        # Load the images if we want to cache them in memory.
        if self.memcache:
            for i, img_filename in enumerate(self.image_filenames):
                img_lr = np.array(Image.open(os.path.join(self.lr_dir, img_filename)).convert("RGB")).astype(np.uint8)
                img_lr = (img_lr / 127.5) - 1.0
                img_lr = img_lr.transpose(2, 0, 1).astype(np.float32)
                self.image_lr[i] = img_lr

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, i):
        img_filename = self.image_filenames[i]
        if self.memcache:
            img_lr = self.image_lr[i]
        else:
            img_lr = np.array(Image.open(os.path.join(self.lr_dir, img_filename)).convert("RGB")).astype(np.uint8)
            img_lr = (img_lr / 127.5) - 1.0
            img_lr = img_lr.transpose(2, 0, 1).astype(np.float32)
        return {
            'img_filename': img_filename,
            'img_lr': img_lr
        }


class Crop_LR_GT_PairTransform(object):
    def __init__(self, scale, patch_size):
        self.scale = scale
        self.patch_size = patch_size

    def __call__(self, sample):
        img_lr, img_gt = sample[0], sample[1]
        ih, iw = img_lr.shape[:2]
        ix = random.randrange(0, iw - self.patch_size + 1)
        iy = random.randrange(0, ih - self.patch_size + 1)
        tx = ix * self.scale
        ty = iy * self.scale
        patch_lr = img_lr[iy: iy + self.patch_size, ix: ix + self.patch_size]
        patch_gt = img_gt[ty: ty + (self.scale * self.patch_size), tx: tx + (self.scale * self.patch_size)]
        return patch_lr, patch_gt


class Random_LR_GT_AugmentationTransform(object):
    def __call__(self, sample):
        img_lr, img_gt = sample[0], sample[1]
        if bool(random.getrandbits(1)):
            img_lr = np.fliplr(img_lr).copy()
            img_gt = np.fliplr(img_gt).copy()
        if bool(random.getrandbits(1)):
            img_lr = np.flipud(img_lr).copy()
            img_gt = np.flipud(img_gt).copy()
        if bool(random.getrandbits(1)):
            img_lr = img_lr.transpose(1, 0, 2)
            img_gt = img_gt.transpose(1, 0, 2)
        return img_lr, img_gt
