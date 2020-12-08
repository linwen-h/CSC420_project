from pathlib import Path
import functools

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', 'tiff']


def is_image(path: Path):
    return path.suffix.lower() in IMG_EXTENSIONS


