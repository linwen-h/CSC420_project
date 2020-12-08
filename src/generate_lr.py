import os
import argparse
import logging
from pathlib import Path

from PIL import Image
from util import arg_util

__description__ = "Generates scaled down low-resolution images from a flat folder of high-resolution images."


def down_sample(img_filepath: Path, scale: int = 4):
    if not img_filepath.is_file():
        return None
    img_hr = Image.open(img_filepath)
    if img_hr.mode != 'RGB':
        img_hr_rgb = Image.new('RGB', img_hr.size, 'WHITE')
        img_hr_rgb.paste(img_hr, (0, 0), img_hr)
    else:
        img_hr_rgb = img_hr
    return img_hr_rgb.resize((int(img_hr_rgb.width // scale), int(img_hr_rgb.height // scale)), Image.ANTIALIAS)


def main(args):
    input_path = args.input_path
    output_path = args.output_path
    if input_path == output_path and not args.force:
        raise ValueError("If input_path and output_path are intentionally equal, --force option must be used.")

    if input_path.is_file():
        img = down_sample(input_path, args.scale)
        if output_path.is_dir():
            img.save(output_path / input_path.name)
        else:
            img.save(output_path)
    elif input_path.is_dir():
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        elif not output_path.is_dir():
            raise ValueError("If input_path is dir, output_path must be dir.")
        for input_file_path in input_path.glob("*"):
            try:
                logging.info(f"Processing: {input_file_path}")
                img = down_sample(input_file_path, args.scale)
                img.save(output_path / input_file_path.name)
            except:
                logging.exception(f"Failed to down-sample: {input_file_path}")
    else:
        return -1
    return 0


def init_args(parser):
    parser.add_argument('--input_path', type=arg_util.path_abs, required=True, help="TODO")
    parser.add_argument('--output_path', type=arg_util.path_abs, required=True, help="TODO")
    parser.add_argument('--force', action='store_true', default=False, help="TODO")
    parser.add_argument('--scale', type=int, default=4)
    return parser


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
                        handlers=[logging.StreamHandler(stream=sys.stdout)])
    logging.info(f"Starting {__file__}")
    logging.info(f"PYTHONPATH=\"{os.environ['PYTHONPATH'] if 'PYTHONPATH' in os.environ else ''}\"")
    logging.info(f"{sys.executable} {' '.join(sys.argv)}")
    exit(main(init_args(argparse.ArgumentParser(description=__description__)).parse_args()))
