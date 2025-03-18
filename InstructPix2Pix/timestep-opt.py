from __future__ import annotations

import time
from itertools import product
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import os
from pathlib import Path
import math
import json
from tqdm import tqdm
import sys
from argparse import ArgumentParser
from glob import glob

import einops
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast
from torchvision.transforms import ToPILImage, ToTensor
from os.path import join

sys.path.append("./stable_diffusion")
# from stable_diffusion.ldm.util import instantiate_from_config
from ldm.util import instantiate_from_config

sys.path.append("./")
from edit_dataset import EditDatasetValid5000


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model."):]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model


def cal_pnsr_img(im1, im2):
    x1 = ToTensor()(im1).permute(2, 1, 0).numpy()
    x2 = ToTensor()(im2).permute(2, 1, 0).numpy()
    psnr = compare_psnr(x1, x2)
    return psnr


def cal_psnr(x1s, x2s):
    psnrs = []
    for x1, x2 in zip(x1s, x2s):
        x1 = x1.detach().cpu()
        x2 = x2.detach().cpu()
        x1 = x1.permute(2, 1, 0).numpy()
        x2 = x2.permute(2, 1, 0).numpy()
        psnr = compare_psnr(x1, x2)
        psnrs.append(psnr)
    psnr_mean = sum(psnrs) / len(psnrs)
    return psnr_mean


def update_p(p, lr):
    assert p > 0
    assert 0 < abs(lr) < 1

    if p == 1:
        if lr > 0:
            return p + lr
        else:
            return 1 / (1 / p - lr)
    elif p > 1:
        if lr > 0:
            return p + lr
        else:
            if p + lr < 1:
                return 1
            return p + lr
    elif p < 1:
        if lr > 0:
            if 1 / p - lr < 1:
                return 1
            return 1 / (1 / p - lr)
        else:
            return 1 / (1 / p - lr)
    else:
        raise NotImplementedError


def timesteps3func_cho(gamma, n, T=999, coef=60):

    s = 1 if gamma >= 1 else -1
    p = gamma if gamma >= 1 else 1 / gamma

    t_l, t_u = 0, T
    t = torch.linspace(t_l, t_u, n)
    t_l = t_l if s == 1 else t_l + coef * (1 - p)
    t_u = t_u if s == -1 else t_u + coef * (p - 1)
    t_ = (t - t_l) / (t_u - t_l)
    t__ = ((t_**gamma).clamp(0, 1) * T).to(torch.int32).flip(0)

    return t__


def save_img(x, path):
    x = x.detach().cpu()
    if len(x.shape) == 4:
        x = x[0]
    ToPILImage()(x).save(path)


def inference(
    model,
    inputs,
    time,
    batch_size=10,
    seed=22,
    eta=0.0,
    cfg_image=1.0,
    cfg_text=7.5,
    max_sample=500,
    offset=0,
):
    null_token = model.get_learned_conditioning([""])

    alphas = model.alphas_cumprod
    sqrt_alphas = model.sqrt_alphas_cumprod
    sqrt_betas = model.sqrt_one_minus_alphas_cumprod

    inputs = DataLoader(inputs, batch_size=batch_size)

    outputs = []
    with model.ema_scope(), torch.no_grad():
        for iter_idx, sample in enumerate(inputs):
            if iter_idx * batch_size < offset:
                continue
            if iter_idx * batch_size >= max_sample + offset:
                break
            input_image = sample["image_0"].to(model.device)
            prompt = sample["edit"]

            cond = {}
            cond["c_crossattn"] = [model.get_learned_conditioning(prompt)]
            cond["c_concat"] = [model.encode_first_stage(input_image).mode()]

            uncond = {}
            uncond["c_crossattn"] = [torch.cat([null_token] * batch_size)]
            uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])] * batch_size

            seed_everything(seed)
            x_t = torch.randn(1, 4, 64, 64).cuda()
            x_t = torch.cat([x_t] * batch_size, dim=0)

            if isinstance(time, int):
                timesteps = torch.linspace(0, 999, time).to(torch.int32).flip(-1)
            else:
                timesteps = sorted(time, reverse=True)

            cnt = len(timesteps)
            for idx in tqdm(range(cnt)):
                t = timesteps[idx]

                ts = torch.full((batch_size, ), t, dtype=torch.long).cuda()

                cfg_z = torch.cat([x_t] * 3, dim=0)
                cfg_ts = torch.cat([ts] * 3, dim=0)

                cfg_cond = {
                    "c_crossattn": [
                        torch.cat([
                            cond["c_crossattn"][0], uncond["c_crossattn"][0],
                            uncond["c_crossattn"][0]
                        ])
                    ],
                    "c_concat":
                    [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
                }
                out_cond, out_img_cond, out_uncond = model.apply_model(cfg_z, cfg_ts,
                                                                       cond=cfg_cond).chunk(3)
                e_t = out_uncond + cfg_text * (out_cond - out_img_cond) + cfg_image * (
                    out_img_cond - out_uncond)

                x_0 = 1 / sqrt_alphas[t] * (x_t - sqrt_betas[t] * e_t)

                if idx == cnt - 1:
                    x = model.decode_first_stage(x_0)
                    x = x.add(1).mul(0.5).clamp(0, 1)
                    outputs.append(x.detach().cpu())
                    break

                tm1 = timesteps[idx + 1]
                sigma = eta * np.sqrt((1 - alphas[tm1].item()) / (1 - alphas[t].item()) *
                                      (1 - alphas[t].item() / alphas[tm1].item()))
                x_t = sqrt_alphas[tm1] * x_0 + torch.sqrt(
                    1 - alphas[tm1] - sigma**2) * e_t + sigma * torch.randn_like(x_t)

        outputs = torch.cat(outputs, dim=0)
    return outputs


@torch.no_grad()
def main():
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=50, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="checkpoints/instruct-pix2pix-00-22000.ckpt", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--input", default="imgs/face5/0030.png", type=str)
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.0, type=float)
    parser.add_argument("--coef", default=30, type=float)
    args = parser.parse_args()

    sys.path.append(os.getcwd())
    cfn = Path(os.path.basename(__file__)).stem
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda()
    seed = 22

    # define logging function
    eta = 0.0
    total_gt = 50
    p_init = 1.0
    coef = args.coef

    dataset = EditDatasetValid5000('data/valid5000', res=512)
    for total in [5, 10, 15, 20]:
        max_sample = 100
        offset = 0
        batch_size = 8

        start_time = time.time()
        lrs = [0.2, 0.1, 0.05, 0.02, 0.01]
        lr = lrs.pop(0)
        dir_log = join("logs", cfn, f"{total:02d}", f"{coef:1.3f}")
        os.makedirs(dir_log, exist_ok=True)

        with open(join(dir_log, "meta.txt"), 'w') as f:
            f.write("# START" + '\n')

        def logging(*txts):
            with open(join(dir_log, "meta.txt"), 'a') as f:
                for txt in txts:
                    f.write(txt + '\n')

        logging(
            f"cfg_text: {args.cfg_text:1.1f}",
            f"cfg_image: {args.cfg_image:1.1f}",
            f"eta: {eta:1.1f}",
            f"total_init: {total:03d}",
            f"p_init: {p_init:1.2f}",
        )

        sign = -1
        p_opt = p_init
        psnr_opt = 0
        psnr = 0
        with autocast("cuda"), model.ema_scope():
            logging("# GET GT Image")
            gt = inference(model,
                           dataset,
                           total_gt,
                           batch_size=batch_size,
                           seed=seed,
                           eta=eta,
                           cfg_image=args.cfg_image,
                           cfg_text=args.cfg_text,
                           max_sample=max_sample,
                           offset=offset)

            logging("# GET Uniform Image")
            uniform = inference(model,
                                dataset,
                                total,
                                batch_size=batch_size,
                                seed=seed,
                                eta=eta,
                                cfg_image=args.cfg_image,
                                cfg_text=args.cfg_text,
                                max_sample=max_sample,
                                offset=offset)

            psnr = cal_psnr(gt, uniform)
            logging(f"total:{total:02d},psnr:{psnr:2.4f},p:{p_opt:2.4f}", )

            while True:
                p = update_p(p_opt, sign * lr)
                t = timesteps3func_cho(p, total, coef=coef)
                x = inference(model,
                              dataset,
                              t,
                              batch_size=batch_size,
                              seed=seed,
                              eta=eta,
                              cfg_image=args.cfg_image,
                              cfg_text=args.cfg_text,
                              max_sample=max_sample,
                              offset=offset)
                psnr = cal_psnr(gt, x)
                if psnr > psnr_opt:
                    psnr_opt = psnr
                    p_opt = p

                    logging(f"total:{total:02d},psnr:{psnr_opt:2.4f},p:{p_opt:2.4f}", )
                else:
                    if len(lrs) == 0:
                        t_opt = timesteps3func_cho(p_opt, total, coef=coef)
                        logging(
                            "# FIND BEST",
                            f"total:{total:02d},psnr:{psnr_opt:2.4f},p:{p_opt:2.4f},coef:{coef:2.2f}",
                        )
                        break
                    else:
                        lr = lrs.pop(0)
            end_time = time.time()
            logging(f'Elapsed time: {end_time - start_time} sec')
            logging(f'Optimal sequence: {t_opt.tolist()} sec')

            # VALIDATION STAGE
            logging('Start Validation')
            offset = max_sample
            max_sample = 1000
            gt = inference(model,
                           dataset,
                           total_gt,
                           batch_size=batch_size,
                           seed=seed,
                           eta=eta,
                           cfg_image=args.cfg_image,
                           cfg_text=args.cfg_text,
                           max_sample=max_sample,
                           offset=offset)
            x = inference(model,
                          dataset,
                          t_opt,
                          batch_size=batch_size,
                          seed=seed,
                          eta=eta,
                          cfg_image=args.cfg_image,
                          cfg_text=args.cfg_text,
                          max_sample=max_sample,
                          offset=offset)
            psnr = cal_psnr(gt, x)
            logging(f"psnr:{psnr:2.4f}, sample {max_sample} EA", )


if __name__ == "__main__":
    main()
