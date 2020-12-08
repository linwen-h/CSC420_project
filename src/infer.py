import argparse
import logging

from util import arg_util

from torch.utils.data import DataLoader
from arch.srgan_model import Generator
from data.dataset import *


__description__ = "TODO"


def inference_dir(generator_path, input_dir, output_dir, batch_size=1, num_workers=0, model_res_count=16,
                  filename_prefix=""):
    t_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr_dataset = LowResDataSet(input_dir, memcache=False)
    loader = DataLoader(lr_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    generator = Generator(img_feat=3, n_feats=64, kernel_size=3, num_block=model_res_count)
    generator.load_state_dict(torch.load(generator_path, map_location=t_device))
    generator = generator.to(t_device)
    generator.eval()

    logging.info(f"Starting Inference.")
    with torch.no_grad():
        for batch_i, lr_datum in enumerate(loader):
            img_filenames, img_lrs = lr_datum['img_filename'], lr_datum['img_lr'].to(t_device)
            img_hr_predictions, _ = generator(img_lrs)
            for i in range(len(img_filenames)):  # Independent of batch size.
                img_filename = img_filenames[i]
                output = ((np.clip(img_hr_predictions[i].cpu().numpy(), -1., 1.) + 1.) / 2.).transpose(1, 2, 0)
                result = Image.fromarray((output * 255.).astype(np.uint8))
                result.save(output_dir / f"hr_{img_filename}")
                logging.info(f"Inference Output: {output_dir / f'{filename_prefix}{img_filename}'}")


def main(args):
    input_path = args.input_path
    output_path = args.output_path
    if input_path == output_path:
        raise ValueError("input_path and output_path must be different!")
    if not input_path.exists():
        raise IOError("input_path must be exist!")

    if input_path.is_file():
        raise NotImplementedError() # TODO
    elif input_path.is_dir():
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        elif not output_path.is_dir():
            raise ValueError("If input_path is dir, output_path must be dir.")
        inference_dir(args.generator_path, input_path, output_path,
                      batch_size=args.batch_size,
                      num_workers=args.num_workers,
                      model_res_count=args.model_res_count)
    else:
        return 1
    return 0


def init_args(parser):
    parser.add_argument('--generator_path', type=arg_util.path_abs, required=True, help="TODO")
    parser.add_argument('--input_path', type=arg_util.path_abs, required=True, help="TODO")
    parser.add_argument('--output_path', type=arg_util.path_abs, required=True, help="TODO")

    parser.add_argument('--batch_size', type=int, default=1, help="TODO")
    parser.add_argument('--num_workers', type=int, default=0, help="TODO")
    parser.add_argument('--model_res_count', type=int, default=16, help="TODO")
    return parser


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s", handlers=[logging.StreamHandler(stream=sys.stdout)])
    logging.info(f"Starting {__file__}")
    logging.info(f"PYTHONPATH=\"{os.environ['PYTHONPATH'] if 'PYTHONPATH' in os.environ else ''}\"")
    logging.info(f"{sys.executable} {' '.join(sys.argv)}")
    exit(main(init_args(argparse.ArgumentParser(description=__description__)).parse_args()))
