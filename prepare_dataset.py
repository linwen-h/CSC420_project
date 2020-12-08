import os
import argparse
from skimage.measure import block_reduce
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter

__description__ = "TODO"

def downsample(imgPath):
    if not imgPath.is_file():
        return None

    img = Image.open(imgPath)
    rgbImage = Image.new("RGB", img.size, "WHITE")
    rgbImage.paste(img, (0,0), img)
    w, h = rgbImage.width, rgbImage.height
    img_lr = rgbImage.resize((w//4, h//4), Image.ANTIALIAS)
    
    return img_lr

def main(args):
    inputPath = Path(args.input_path)
    outputPath = Path(args.output_path)

    if not outputPath.exists():
        os.mkdir(outputPath.name)

    if inputPath.is_file():
        img = downsample(inputPath)

        img.save((os.path.join(outputPath, ("lr_" + inputPath.name))))

    elif inputPath.is_dir():
        for p in inputPath.rglob("*"):
            img = downsample(p)
            if img:
                if not Path(os.path.join(outputPath, p.parents[0].name)).exists():
                    os.mkdir(os.path.join(outputPath, p.parents[0].name))
  
                img.save((os.path.join(outputPath, p.parents[0].name, ("lr_" + p.name))))
    else:
        return -1

    
    return 0

def init_args(parser):
    parser.add_argument('-input_path', type=os.path.abspath, help="TODO")
    parser.add_argument('-output_path', type=os.path.abspath, help="TODO")
    # Example: parser.add_argument('--option_A', type=os.path.abspath, help="TODO")
    return parser

if __name__ == '__main__':
    exit(main(init_args(argparse.ArgumentParser(description=__description__)).parse_args()))