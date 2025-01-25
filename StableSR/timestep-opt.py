"""make variations of input image"""

import argparse, os, sys, glob
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import PIL
from pathlib import Path
import torch
import numpy as np
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything
from os.path import join
from tqdm import tqdm

from ldm.util import instantiate_from_config
from scripts.wavelet_color_fix import adaptive_instance_normalization


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def cal_pnsr(x1s, x2s):
    vals = []
    for x1, x2 in zip(x1s, x2s):
        x1 = x1.permute(2, 1, 0).numpy()
        x2 = x2.permute(2, 1, 0).numpy()
        psnr = compare_psnr(x1, x2)
        vals.append(psnr)
    psnr = sum(vals) / len(vals)

    return psnr


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


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image",
        default="data/DIV2K_V2_val_resort/lq",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="logs",
    )
    parser.add_argument(
        "--ddpm_steps",
        type=int,
        default=200,
        help="number of ddpm sampling steps",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stableSRNew/v2-finetune_text_T_512.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="ckpts/stablesr_000117.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--vqgan_ckpt",
        type=str,
        default="ckpts/vqgan_cfw_00011.ckpt",
        help="path to checkpoint of VQGAN model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=22,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument("--precision",
                        type=str,
                        help="evaluate at this precision",
                        choices=["full", "autocast"],
                        default="autocast")
    parser.add_argument(
        "--input_size",
        type=int,
        default=512,
        help="input size",
    )
    parser.add_argument(
        "--dec_w",
        type=float,
        default=0.5,
        help="weight for combining VQGAN and Diffusion",
    )
    parser.add_argument(
        "--coef",
        type=float,
        default=30,
        help="weight for combining VQGAN and Diffusion",
    )
    parser.add_argument(
        "--colorfix_type",
        type=str,
        default="adain",
        help=
        "Color fix type to adjust the color of HR result according to LR input: adain (used in paper); wavelet; nofix",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="weight for combining VQGAN and Diffusion",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="weight for combining VQGAN and Diffusion",
    )

    opt = parser.parse_args()

    # SET TARGET OUTPUT
    cfn = Path(os.path.basename(__file__)).stem
    opt.outdir = os.path.join("logs", cfn)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print('>>>>>>>>>>color correction>>>>>>>>>>>')
    if opt.colorfix_type == 'adain':
        print('Use adain color correction')
    elif opt.colorfix_type == 'wavelet':
        print('Use wavelet color correction')
    else:
        print('No color correction')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    vqgan_config = OmegaConf.load("configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml")
    vq_model = load_model_from_config(vqgan_config, opt.vqgan_ckpt)
    vq_model = vq_model.to(device)
    vq_model.decoder.fusion_w = opt.dec_w

    seed_everything(opt.seed)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(opt.input_size),
        torchvision.transforms.CenterCrop(opt.input_size),
    ])

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    model = model.to(device)

    os.makedirs(opt.outdir, exist_ok=True)

    img_list_ori = sorted(os.listdir(opt.init_img))

    model.register_schedule(given_betas=None,
                            beta_schedule="linear",
                            timesteps=1000,
                            linear_start=0.00085,
                            linear_end=0.0120,
                            cosine_s=8e-3)
    model.num_timesteps = 1000
    model = model.to(device)

    eta = opt.eta
    seed = opt.seed
    batch_size = opt.batch_size
    coef = opt.coef

    alphas = model.alphas_cumprod
    sqrt_alphas = model.sqrt_alphas_cumprod
    sqrt_betas = model.sqrt_one_minus_alphas_cumprod
    structure_stage_model = model.structcond_stage_model
    precision_scope = autocast if opt.precision == "autocast" else nullcontext

    def inference(seed, input_image, time, use_adain=True):
        batch_size = input_image.shape[0]
        text_cond = model.cond_stage_model([''] * batch_size)

        seed_everything(seed)
        x_t = torch.randn(1, 4, 64, 64).cuda()
        x_t = torch.cat([x_t] * batch_size, dim=0)

        if isinstance(time, int):
            timesteps = torch.linspace(0, 999, time).to(torch.int32).flip(-1)
        else:
            timesteps = time

        init_latent_generator, enc_fea_lq = vq_model.encode(input_image)
        input_latent = model.get_first_stage_encoding(init_latent_generator)
        cnt = len(timesteps)
        for idx in tqdm(range(cnt)):
            t = timesteps[idx]
            ts = torch.full((batch_size, ), t, dtype=torch.long).cuda()

            struct_cond = structure_stage_model(input_latent, ts)
            e_t = model.apply_model(x_t, ts, text_cond, struct_cond)
            x_0 = 1 / sqrt_alphas[t] * (x_t - sqrt_betas[t] * e_t)

            if idx == cnt - 1:
                x = model.decode_first_stage(x_0)
                if use_adain:
                    x = adaptive_instance_normalization(x, input_image)
                x = x.add(1).mul(0.5).clamp(0, 1)
                return x

            tm1 = timesteps[idx + 1]
            sigma = eta * np.sqrt((1 - alphas[tm1].item()) / (1 - alphas[t].item()) *
                                  (1 - alphas[t].item() / alphas[tm1].item()))
            x_t = sqrt_alphas[tm1] * x_0 + torch.sqrt(
                1 - alphas[tm1] - sigma**2) * e_t + sigma * torch.randn_like(x_t)

    t_gt = torch.linspace(0, 999, 50).to(torch.int32).flip(-1)


    with torch.no_grad(), precision_scope("cuda"), model.ema_scope():
        for t_total in [5, 10, 15, 20, 25]:
            max_samples = 100
            img_list_ori_chunk = list(divide_chunks(img_list_ori[:max_samples], batch_size))
            start_time = time.time()
            # batch
            dir_log = join("logs", cfn, f"{t_total:02d}", f"{coef:1.3f}")
            os.makedirs(dir_log, exist_ok=True)

            with open(join(dir_log, "meta.txt"), 'w') as f:
                f.write("# START" + '\n')

            def logging(*txts):
                with open(join(dir_log, "meta.txt"), 'a') as f:
                    for txt in txts:
                        f.write(txt + '\n')

            lrs = [0.2, 0.1, 0.05, 0.02, 0.01]
            lr = lrs.pop(0)
            psnr_opt = 0
            p_opt = 1
            logging(f"eta: {eta}", f"total:{t_total}")

            # GET GT
            x_gts = []
            x_uniform_cmps = []

            for paths in img_list_ori_chunk:
                inputs = []
                for path in paths:
                    input_image = load_img(os.path.join(opt.init_img, path)).to(device)
                    input_image = transform(input_image)
                    input_image = input_image.clamp(-1, 1)
                    inputs.append(input_image)
                input_image = torch.cat(inputs, dim=0)
                x_gt = inference(seed, input_image, t_gt)
                x_gts.append(x_gt.detach().cpu())

                # GET UNIFORM COMP
                t_uniform = torch.linspace(0, 999, t_total).to(torch.int32).flip(-1)
                x_uni = inference(seed, input_image, t_uniform)
                x_uniform_cmps.append(x_uni.detach().cpu())

            x_gts = torch.cat(x_gts, dim=0)
            x_uniform_cmps = torch.cat(x_uniform_cmps, dim=0)
            psnr = cal_pnsr(x_gts, x_uniform_cmps)
            logging(f"psnr:{psnr:2.4f},p:{1:2.4f}", )

            sign = 1
            # DIRECTIONAL SEARCH
            while True:
                xs = []
                for paths in img_list_ori_chunk:
                    inputs = []
                    for path in paths:
                        input_image = load_img(os.path.join(opt.init_img, path)).to(device)
                        input_image = transform(input_image)
                        input_image = input_image.clamp(-1, 1)
                        inputs.append(input_image)
                    input_image = torch.cat(inputs, dim=0)
                    p = update_p(p_opt, sign * lr)
                    t = timesteps3func_cho(p, t_total, coef=coef)
                    x = inference(seed, input_image, t)
                    xs.append(x.detach().cpu())

                xs = torch.cat(xs, dim=0)
                psnr = cal_pnsr(x_gts, xs)
                if psnr > psnr_opt:
                    psnr_opt = psnr
                    p_opt = p
                    logging(f"psnr:{psnr_opt:2.4f},p:{p_opt:2.4f}", )
                else:
                    if len(lrs) == 0:
                        t_opt = timesteps3func_cho(p_opt, t_total, coef=coef)
                        logging(
                            "Best",
                            f"psnr:{psnr_opt:2.4f},p:{p_opt:2.4f}",
                        )
                        break
                    else:
                        lr = lrs.pop(0)
            end_time = time.time()
            logging(f'Elapsed time: {end_time - start_time} sec')
            logging(f'Optimal sequence: {t_opt.tolist()} sec')

            # START VALIDATION
            logging('Start Validation')
            x_gts = []
            x_ours = []
            t_our = timesteps3func_cho(p_opt, t_total, coef=coef)
            offset = max_samples
            max_samples = 1000

            img_list_ori_chunk = list(
                divide_chunks(img_list_ori[offset:offset + max_samples], batch_size))
            for paths in img_list_ori_chunk:
                inputs = []
                for path in paths:
                    input_image = load_img(os.path.join(opt.init_img, path)).to(device)
                    input_image = transform(input_image)
                    input_image = input_image.clamp(-1, 1)
                    inputs.append(input_image)
                input_image = torch.cat(inputs, dim=0)
                x_gt = inference(seed, input_image, t_gt)
                x_gts.append(x_gt.detach().cpu())

                x_our = inference(seed, input_image, t_our)
                x_ours.append(x_our.detach().cpu())

            x_gts = torch.cat(x_gts, dim=0)
            x_ours = torch.cat(x_ours, dim=0)
            psnr = cal_pnsr(x_gts, x_ours)
            logging(f"psnr:{psnr:2.4f},sample {max_samples} EA", )


if __name__ == "__main__":
    main()
