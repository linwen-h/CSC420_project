import argparse
import logging

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from arch.srgan_model import Generator, Discriminator
from arch.vgg19 import vgg19
from data.dataset import *
from arch.losses import TVLoss, perceptual_loss
from util import arg_util


__description__ = "TODO"




def train(generator_path_out, discriminator_path_out, checkpoint_dir, gt_path, lr_path,
          memcache=True, batch_size=32, num_workers=0,
          scale=4, patch_size=24, model_res_count=16,
          transfer_generator_path=None,
          pre_train_epoch=40, adversarial_train_epoch=100,
          feat_layer='relu5_4', vgg_rescale_coeff=0.006, adv_coeff=1e-3, tv_loss_coeff=0.0):
    if pre_train_epoch <= 0 and adversarial_train_epoch <= 0:
        logging.info(f"Nothing to do: pre_train_epoch <= 0 and adversarial_train_epoch <= 0!")
        return

    t_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr_gt_dataset = LowResGroundTruthDataset(lr_dir=lr_path, gt_dir=gt_path, memcache=memcache,
                                             transform=transforms.Compose([
                                                 Crop_LR_GT_PairTransform(scale, patch_size),
                                                 Random_LR_GT_AugmentationTransform()
                                             ]))
    loader = DataLoader(lr_gt_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    generator = Generator(img_feat=3, n_feats=64, kernel_size=3, num_block=model_res_count, scale=scale)
    if transfer_generator_path:
        generator.load_state_dict(torch.load(transfer_generator_path, map_location=t_device))
        logging.info(f"Loaded pre-trained model: {transfer_generator_path}")
    generator = generator.to(t_device)
    generator.train()

    L2_MSE_loss = nn.MSELoss()
    g_optim = optim.Adam(generator.parameters(), lr=1e-4)

    # Pre-train using L2 loss
    if pre_train_epoch > 0:
        logging.info(f"Pre-training using L2 loss for {pre_train_epoch} epochs.")
        checkpoint_modulo = (pre_train_epoch // 10) or pre_train_epoch
        for pre_epoch in range(1, pre_train_epoch + 1):
            logging.info(f"Pre-train Epoch [{pre_epoch}]: running.")
            for batch_i, lr_gt_datum in enumerate(loader):
                img_lr, img_gt = lr_gt_datum['img_lr'].to(t_device), lr_gt_datum['img_gt'].to(t_device)
                img_hr_prediction, _ = generator(img_lr)
                loss = L2_MSE_loss(img_hr_prediction, img_gt)
                g_optim.zero_grad()
                loss.backward()
                g_optim.step()

            # Log epoch statistics.
            logging.info(f"Pre-train Epoch [{pre_epoch}]: loss={loss.item()}")
            if pre_epoch % checkpoint_modulo == 0:
                checkpoint_filepath = (checkpoint_dir / f'pre_trained_model_{pre_epoch}.pt').absolute()
                torch.save(generator.state_dict(),  checkpoint_filepath)
                logging.info(f"Pre-train Epoch [{pre_epoch}]: saved model checkpoint: {checkpoint_filepath}")

    # Train using perceptual & adversarial loss
    if adversarial_train_epoch > 0:
        logging.info(f"Training using Adversarial loss for {adversarial_train_epoch} epochs.")

        # Set-up adversarial loss VGG network.
        vgg_net = vgg19().to(t_device)
        vgg_net = vgg_net.eval()

        discriminator = Discriminator(patch_size=patch_size * scale)
        discriminator = discriminator.to(t_device)
        discriminator.train()

        d_optim = optim.Adam(discriminator.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(g_optim, step_size=2000, gamma=0.1)

        VGG_loss = perceptual_loss(vgg_net)
        cross_ent = nn.BCELoss()
        tv_loss = TVLoss()
        base_real_label = torch.ones((batch_size, 1)).to(t_device)
        base_fake_label = torch.zeros((batch_size, 1)).to(t_device)

        torch.autograd.set_detect_anomaly(True)
        checkpoint_modulo = (adversarial_train_epoch // 10) or adversarial_train_epoch
        for epoch in range(1, adversarial_train_epoch + 1):
            logging.info(f"Epoch [{epoch}]: running.")

            d_optim.step()
            g_optim.step()
            scheduler.step()
            for batch_i, lr_gt_datum in enumerate(loader):
                img_lr, img_gt = lr_gt_datum['img_lr'].to(t_device), lr_gt_datum['img_gt'].to(t_device)
                img_hr_prediction, _ = generator(img_lr)

                # Train Discriminator
                fake_prob = discriminator(img_hr_prediction)
                real_prob = discriminator(img_gt)

                # Avoid mismatched label and probability length in case where batch is remainder of data, but not
                # a perfect fit.
                real_label = base_real_label
                fake_label = base_fake_label
                if len(base_real_label) != len(real_prob):
                    real_label = torch.ones((len(real_prob), 1)).to(t_device)
                    fake_label = torch.zeros((len(real_prob), 1)).to(t_device)

                d_loss_real = cross_ent(real_prob, real_label)
                d_loss_fake = cross_ent(fake_prob, fake_label)

                d_loss = d_loss_real + d_loss_fake

                # Back-propagate Discriminator
                g_optim.zero_grad()
                d_optim.zero_grad()
                d_loss.backward()
                d_optim.step()

                # Train Generator
                img_hr_prediction, _ = generator(img_lr)
                fake_prob = discriminator(img_hr_prediction)

                l2_loss = L2_MSE_loss(img_hr_prediction, img_gt)
                percep_loss, hr_feat, sr_feat = VGG_loss((img_gt + 1.0) / 2.0, (img_hr_prediction + 1.0) / 2.0, layer=feat_layer)
                percep_loss = vgg_rescale_coeff * percep_loss
                adversarial_loss = adv_coeff * cross_ent(fake_prob, real_label)
                total_variance_loss = tv_loss_coeff * tv_loss(vgg_rescale_coeff * (hr_feat - sr_feat) ** 2)
                g_loss = percep_loss + adversarial_loss + total_variance_loss + l2_loss

                # Back-propagate Generator
                g_optim.zero_grad()
                d_optim.zero_grad()
                g_loss.backward()
                g_optim.step()

            # Log epoch statistics.
            logging.info(f"Epoch [{epoch}]: g_loss={g_loss.item()} d_loss={d_loss.item()}")
            if epoch % checkpoint_modulo == 0:
                g_checkpoint_filepath = (checkpoint_dir / f'SRGAN_g_{epoch}.pt').absolute()
                d_checkpoint_filepath = (checkpoint_dir / f'SRGAN_d_{epoch}.pt').absolute()
                torch.save(generator.state_dict(),  g_checkpoint_filepath)
                torch.save(discriminator.state_dict(), d_checkpoint_filepath)
                logging.info(f"Pre-train Epoch [{epoch}]: saved model checkpoints: {g_checkpoint_filepath}, {d_checkpoint_filepath}")
        if discriminator_path_out:
            torch.save(discriminator.state_dict(), discriminator_path_out)
    torch.save(generator.state_dict(), generator_path_out)


def main(args):
    generator_path_out = args.generator_path_out
    generator_path_out.parent.mkdir(parents=True, exist_ok=True)

    discriminator_path_out = args.discriminator_path_out
    discriminator_path_out.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)


    gt_path = args.gt_path
    lr_path = args.lr_path
    if gt_path == lr_path:
        raise ValueError("gt_path, lr_path must be different!")

    if not gt_path.exists() or not gt_path.is_dir():
        raise IOError("gt_path must be an existing directory!")
    if not lr_path.exists() or not lr_path.is_dir():
        raise IOError("lr_path must be an existing directory!")

    train(
        generator_path_out=generator_path_out,
        discriminator_path_out=discriminator_path_out,
        checkpoint_dir=checkpoint_dir,
        gt_path=gt_path,
        lr_path=lr_path,

        memcache=args.memcache_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,

        scale=args.scale,
        patch_size=args.patch_size,
        model_res_count=args.model_res_count,

        transfer_generator_path=args.transfer_generator_path,
        pre_train_epoch=args.pre_train_epoch,
        adversarial_train_epoch=args.adversarial_train_epoch,

        feat_layer='relu5_4',
        vgg_rescale_coeff=0.006,
        adv_coeff=1e-3,
        tv_loss_coeff=0.0)
    return 0


""" Example args:
--generator_path_out ../train_out/SRGAN_g.pt
--discriminator_path_out ../train_out/SRGAN_d.pt
--checkpoint_dir ../train_out/
--gt_path ../data/pokemon/hr
--lr_path ../data/pokemon/lr
--transfer_generator_path ../pretrained/SRGAN.pt
"""
def init_args(parser):
    parser.add_argument('--generator_path_out', type=arg_util.path_abs, required=True, help="TODO")
    parser.add_argument('--discriminator_path_out', type=arg_util.path_abs, required=True, help="TODO")
    parser.add_argument('--checkpoint_dir', type=arg_util.path_abs, required=True, help="TODO")

    parser.add_argument('--gt_path', type=arg_util.path_abs, required=True, help="TODO")
    parser.add_argument('--lr_path', type=arg_util.path_abs, required=True, help="TODO")

    parser.add_argument('--memcache_data', type=arg_util.str2bool, default=False, help="TODO")
    parser.add_argument('--batch_size', type=int, default=32, help="TODO")
    parser.add_argument('--num_workers', type=int, default=0, help="TODO")

    parser.add_argument('--scale', type=int, default=4, help="TODO")
    parser.add_argument('--patch_size', type=int, default=24, help="TODO")
    parser.add_argument('--model_res_count', type=int, default=16, help="TODO")

    parser.add_argument('--transfer_generator_path', type=arg_util.path_abs, required=True, help="TODO")

    parser.add_argument('--pre_train_epoch', type=int, default=40, help="TODO")
    parser.add_argument('--adversarial_train_epoch', type=int, default=100, help="TODO")
    return parser


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s", handlers=[logging.StreamHandler(stream=sys.stdout)])
    logging.info(f"Starting {__file__}")
    logging.info(f"PYTHONPATH=\"{os.environ['PYTHONPATH'] if 'PYTHONPATH' in os.environ else ''}\"")
    logging.info(f"{sys.executable} {' '.join(sys.argv)}")
    exit(main(init_args(argparse.ArgumentParser(description=__description__)).parse_args()))
