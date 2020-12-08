import argparse
import logging
import csv

from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio
from torch.utils.data import DataLoader

from arch.srgan_model import Generator
from data.dataset import *
from util import arg_util

__description__ = "TODO"


def test(generator_path, gt_path, lr_path, output_path, psnr_result_filepath,
         memcache=True, batch_size=16, num_workers=0, scale=4, model_res_count=16, filename_prefix=""):
    t_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr_gt_dataset = LowResGroundTruthDataset(lr_dir=lr_path, gt_dir=gt_path, memcache=memcache, transform=None)
    loader = DataLoader(lr_gt_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    generator = Generator(img_feat=3, n_feats=64, kernel_size=3, num_block=model_res_count)
    generator.load_state_dict(torch.load(generator_path, map_location=t_device))
    generator = generator.to(t_device)
    generator.eval()

    logging.info(f"Starting Testing.")
    with torch.no_grad():
        psnr_data = []
        for batch_i, lr_gt_datum in enumerate(loader):
            img_filenames, img_lrs = lr_gt_datum['img_filename'], lr_gt_datum['img_lr'].to(t_device)
            img_hr_predictions, _ = generator(img_lrs)
            img_gts = lr_gt_datum['img_gt'].to(t_device)
            for i in range(len(img_filenames)):  # Independent of batch size.
                img_filename = img_filenames[i]

                img_gt = ((img_gts[i].cpu().numpy() + 1.) / 2.).transpose(1, 2, 0)
                img_hr_prediction = ((np.clip(img_hr_predictions[i].cpu().numpy(), -1., 1.) + 1.) / 2.).transpose(1, 2, 0)

                # Since no transforms are used, resize GT to ensure its the same as HR.
                img_gt = img_gt[:img_hr_prediction.shape[0], :img_hr_prediction.shape[1], :]

                # Calculate psnr from ycbcr comparison.
                y_hr_prediction = rgb2ycbcr(img_hr_prediction)[scale:-scale, scale:-scale, :1]
                y_gt = rgb2ycbcr(img_gt)[scale:-scale, scale:-scale, :1]
                psnr = peak_signal_noise_ratio(y_gt / 255., y_hr_prediction / 255., data_range=1.)
                psnr_data.append({'img_filepath': str(lr_gt_dataset.lr_dir / img_filename), 'psnr': psnr})
                logging.info(f"{img_filename}: psnr={psnr}")

                result = Image.fromarray((img_hr_prediction * 255.).astype(np.uint8))
                result.save(output_path / f"gt_{img_filename}")
                logging.info(f"Inference Output: {output_path / f'{filename_prefix}{img_filename}'}")

        if psnr_result_filepath:
            with open(psnr_result_filepath, 'w') as fp_psrn_results:
                writer = csv.DictWriter(fp_psrn_results, fieldnames=['img_filepath', 'psnr'])
                writer.writeheader()
                writer.writerows(psnr_data)


def main(args):
    generator_path = args.generator_path
    if not generator_path.exists():
        raise IOError("generator_path must exist!")
    gt_path = args.gt_path
    lr_path = args.lr_path
    output_path = args.output_path
    if gt_path == output_path or gt_path == lr_path or lr_path == output_path:
        raise ValueError("gt_path, lr_path, output_path must be different!")
    if not gt_path.exists() or not gt_path.is_dir():
        raise IOError("gt_path must be an existing directory!")
    if not lr_path.exists() or not lr_path.is_dir():
        raise IOError("lr_path must be an existing directory!")
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    elif not output_path.is_dir():
        raise ValueError("If input_path is dir, output_path must be dir.")

    test(generator_path=generator_path,
         gt_path=gt_path,
         lr_path=lr_path,
         output_path=output_path,
         psnr_result_filepath=args.psnr_result_filepath,
         memcache=args.memcache_data,
         batch_size=args.batch_size,
         num_workers=args.num_workers,
         scale=args.scale,
         model_res_count=args.model_res_count)
    return 0


def init_args(parser):
    parser.add_argument('--generator_path', type=arg_util.path_abs, required=True, help="TODO")
    parser.add_argument('--gt_path', type=arg_util.path_abs, required=True, help="TODO")
    parser.add_argument('--lr_path', type=arg_util.path_abs, required=True, help="TODO")
    parser.add_argument('--output_path', type=arg_util.path_abs, required=True, help="TODO")
    parser.add_argument('--psnr_result_filepath', type=arg_util.path_abs, required=False, help="TODO")

    parser.add_argument('--memcache_data', type=arg_util.str2bool, default=False, help="TODO")
    parser.add_argument('--batch_size', type=int, default=16, help="TODO")
    parser.add_argument('--num_workers', type=int, default=0, help="TODO")

    parser.add_argument('--scale', type=int, default=4, help="TODO")
    parser.add_argument('--model_res_count', type=int, default=16, help="TODO")
    return parser


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s", handlers=[logging.StreamHandler(stream=sys.stdout)])
    logging.info(f"Starting {__file__}")
    logging.info(f"PYTHONPATH=\"{os.environ['PYTHONPATH'] if 'PYTHONPATH' in os.environ else ''}\"")
    logging.info(f"{sys.executable} {' '.join(sys.argv)}")
    exit(main(init_args(argparse.ArgumentParser(description=__description__)).parse_args()))
