import cv2
import csv
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio
from torch.utils.data import DataLoader
from arch.srgan_model import Generator, Discriminator
from arch.vgg19 import vgg19
from arch.losses import TVLoss, perceptual_loss
from arch import arg_util
from arch.dataset import *
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import multiprocessing
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np

t_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MetricEval(object):
    """Object for evaluating metrics"""

    def __init__(self, train_dataset, memcache=True, transform=None, batch_size=24, num_workers=multiprocessing.cpu_count()):
        # Train, Validation, Test is the order of each list
        self.modes = ["train", "val", "test"]
        self.output_paths = [arg_util.path_abs(
            f"{mode}_out/") for mode in self.modes]
        self.lr_paths = [arg_util.path_abs(
            f"data/pokemon/lr/{mode}") for mode in self.modes]
        self.gt_paths = [arg_util.path_abs(
            f"data/pokemon/hr/{mode}") for mode in self.modes]

        dataset_params = {"memcache": memcache, "transform": transform}
        self.datasets = [train_dataset] + [
            LowResGroundTruthDataset(
                lr_dir=lr_path, gt_dir=gt_path, **dataset_params)
            for lr_path, gt_path in zip(self.lr_paths[1:], self.gt_paths[1:])
        ]

        loader_params = {"batch_size": batch_size, "shuffle": False,
                         "drop_last": False, "num_workers": num_workers}
        self.loaders = [DataLoader(dataset, **loader_params)
                        for dataset in self.datasets]

        self.tv_loss = TVLoss()
        self.vgg_net = vgg19().to(t_device)
        self.vgg_net = self.vgg_net.eval()
        self.vgg_loss = perceptual_loss(self.vgg_net)

    def load_generator(self, generator_path=None, generator=None):
        if generator_path is None and generator is None:
            raise ValueError(
                f"One of generator_path or generator must not be None.")

        if generator_path:
            self.generator = Generator(
                img_feat=3, n_feats=64, kernel_size=3, num_block=model_res_count, scale=scale)
            self.generator.load_state_dict(torch.load(
                generator_path, map_location=t_device))
            self.generator = self.generator.to(t_device)
        else:
            self.generator = generator
        self.generator = self.generator.eval()

    def get_metric(self, mode="val", metric="MSE", write_img=False):
        # Valid Metrics: MSE, PSNR, VGG22, VGG54 (TODO: SSIM)
        scale = 4
        patch_size = 24
        model_res_count = 16

        # feat_layer='relu2_2'
        feat_layer = 'relu5_4'
        vgg_rescale_coeff = 0.006
        adv_coeff = 1e-3
        tv_loss_coeff = 0.0

        idx = self.modes.index(mode)
        with torch.no_grad():
            results = []
            for lr_gt_datum in self.loaders[idx]:
                img_filenames = lr_gt_datum['img_filename']
                img_lrs = lr_gt_datum['img_lr'].to(t_device)
                img_gts = lr_gt_datum['img_gt'].to(t_device)

                img_preds, _ = self.generator(img_lrs)

                img_lrs.cpu()
                img_gts.cpu()
                img_preds.cpu()

                # Revert from [-1, 1] -> [0, 1]
                img_gts = ((img_gts + 1.) / 2.)
                img_preds = ((torch.clip(img_preds, -1., 1.) + 1.) / 2.)

                # Resize GT to ensure its the same size as HR.
                img_gts = img_gts[:, :, :img_preds.shape[2],
                                  :img_preds.shape[3]]
                if metric == "MSE":
                    loss = F.mse_loss(img_preds, img_gts)
                    results.append(loss)
                elif metric == "PSNR":
                    # Calculate psnr from ycbcr comparison. (N, H, W, C)
                    y_preds = img_preds.cpu().numpy().transpose(0, 2, 3, 1)
                    y_gt = img_gts.cpu().numpy().transpose(0, 2, 3, 1)

                    y_preds = rgb2ycbcr(y_preds)[
                        :, scale:-scale, scale:-scale, 0]
                    y_gt = rgb2ycbcr(y_gt)[:, scale:-scale, scale:-scale, 0]

                    psnr = peak_signal_noise_ratio(
                        y_gt / 255., y_preds / 255., data_range=1.)
                    results.append(psnr)

                elif metric == "VGG22" or metric == "VGG54":
                    img_gts = img_gts.to(t_device)
                    img_preds = img_preds.to(t_device)

                    feat_layer = "relu2_2" if metric == "VGG22" else "relu5_4"
                    _percep_loss, hr_feat, sr_feat = self.vgg_loss(
                        img_gts, img_preds, layer=feat_layer)

                    g_loss = F.mse_loss(img_preds, img_gts) + \
                        vgg_rescale_coeff * _percep_loss + \
                        tv_loss_coeff * \
                        self.tv_loss(vgg_rescale_coeff *
                                     (hr_feat - sr_feat)**2)
                    results.append(g_loss)

                    img_gts = img_gts.cpu()
                    img_preds = img_preds.cpu()

                if write_img:
                    for i in range(len(img_filenames)):
                        result = Image.fromarray(
                            (img_preds[i] * 255.).permute((1, 2, 0)).to(torch.uint8).cpu().numpy())
                        result.save(
                            self.output_paths[idx] / f"pred_{img_filenames[i]}")
                        logging.info(
                            f"Inference Output: {self.output_paths[idx] / f'pred_{img_filenames[i]}'}")

            print(f"Average {metric} Score: {sum(results)/len(results)}")
        return sum(results)/len(results)

    def save_test_metrics(self, generator, generator2, mode="test"):
        # Valid Metrics: MSE, PSNR, VGG22, VGG54
        idx = self.modes.index(mode)
        image_filenames = self.datasets[idx].image_filenames
        results_fp = arg_util.path_abs("results/")

        scale = 4
        patch_size = 24
        model_res_count = 16

        # feat_layer='relu2_2'
        feat_layer = 'relu5_4'
        vgg_rescale_coeff = 0.006
        adv_coeff = 1e-3
        tv_loss_coeff = 0.0

        with torch.no_grad():
            for i, filename in enumerate(image_filenames):
                img_lr, img_gt = self.datasets[idx].image_lr_gt_pairs[i]

                # Apply Normalization from [0, 1] -> [-1, 1]
                img_lr = (torch.unsqueeze(transforms.ToTensor()
                                          (img_lr), 0).to(t_device) * 2) - 1.0
                img_gt = (torch.unsqueeze(transforms.ToTensor()
                                          (img_gt), 0).to(t_device) * 2) - 1.0

                img_pred, _ = generator(img_lr)
                img_pred = ((torch.clip(img_pred, -1., 1.) + 1.) / 2.)
                img_pred2, _ = generator2(img_lr)
                img_pred2 = ((torch.clip(img_pred2, -1., 1.) + 1.) / 2.)

                img_lr = ((img_lr + 1.) / 2.)
                img_gt = ((img_gt + 1.) / 2.)

                # Resize GT to ensure its the same size as HR.
                img_gt = img_gt[:, :, :img_pred.shape[2], :img_pred.shape[3]]

                # Calculate psnr from ycbcr comparison. (N, H, W, C)
                y_preds = img_pred.cpu().numpy().transpose(0, 2, 3, 1)
                y_preds2 = img_pred2.cpu().numpy().transpose(0, 2, 3, 1)
                y_gt = img_gt.cpu().numpy().transpose(0, 2, 3, 1)

                y_preds = rgb2ycbcr(y_preds)[:, scale:-scale, scale:-scale, 0]
                y_preds2 = rgb2ycbcr(y_preds2)[
                    :, scale:-scale, scale:-scale, 0]
                y_gt = rgb2ycbcr(y_gt)[:, scale:-scale, scale:-scale, 0]

                psnr = peak_signal_noise_ratio(
                    y_gt / 255., y_preds / 255., data_range=1.)
                psnr2 = peak_signal_noise_ratio(
                    y_gt / 255., y_preds2 / 255., data_range=1.)

                img_gt = img_gt.to(t_device)
                img_pred = img_pred.to(t_device)
                img_pred2 = img_pred2.to(t_device)

                _percep_loss, hr_feat, sr_feat = self.vgg_loss(
                    img_gt, img_pred, layer="relu5_4")
                _percep_loss2, hr_feat2, sr_feat2 = self.vgg_loss(
                    img_gt, img_pred2, layer="relu5_4")

                g_loss = F.mse_loss(img_pred, img_gt) + \
                    vgg_rescale_coeff * _percep_loss + \
                    tv_loss_coeff * \
                    self.tv_loss(vgg_rescale_coeff * (hr_feat - sr_feat)**2)
                g_loss2 = F.mse_loss(img_pred2, img_gt) + \
                    vgg_rescale_coeff * _percep_loss2 + \
                    tv_loss_coeff * \
                    self.tv_loss(vgg_rescale_coeff * (hr_feat2 - sr_feat2)**2)

                img_gt = img_gt.cpu()
                img_pred = img_pred.cpu()
                img_pred2 = img_pred2.cpu()

                # Show Results
                img_pred = (img_pred.squeeze() * 255.).permute((1,
                                                                2, 0)).to(torch.uint8).cpu().numpy()
                img_pred2 = (img_pred2.squeeze() * 255.).permute((1,
                                                                  2, 0)).to(torch.uint8).cpu().numpy()
                img_gt = (img_gt.squeeze() * 255.).permute((1, 2, 0)
                                                           ).to(torch.uint8).cpu().numpy()
                img_lr = (img_lr.squeeze() * 255.).permute((1, 2, 0)
                                                           ).to(torch.uint8).cpu().numpy()

                f, ax = plt.subplots(1, 4, figsize=(10., 3.5))
                ax[0].imshow(img_gt)
                ax[0].title.set_text(f"Ground Truth\n")
                ax[0].axis('off')
                ax[1].imshow(img_pred)
                ax[1].title.set_text(
                    f"Prediction\n({np.round(psnr.item(), 2)}dB/{np.round(g_loss.item(), 4)})")
                ax[1].axis('off')
                ax[2].imshow(img_pred2)
                ax[2].title.set_text(
                    f"Prediction (No finetuning)\n({np.round(psnr2.item(), 2)}dB/{np.round(g_loss2.item(), 4)})")
                ax[2].axis('off')
                ax[3].imshow(img_lr)
                ax[3].title.set_text(f"Input\n")
                ax[3].axis('off')

                plt.subplots_adjust(wspace=0, hspace=0,
                                    left=0, right=1, bottom=0, top=1)
                plt.savefig(results_fp / filename, dpi=200)
                plt.clf()
