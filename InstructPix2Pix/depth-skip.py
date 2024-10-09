from __future__ import annotations
from torch.utils.data import DataLoader

from copy import deepcopy
import os
from pathlib import Path
from tqdm import tqdm
import sys
from argparse import ArgumentParser

import einops
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torch import autocast
from torchvision.transforms import ToPILImage
from os.path import join
from pytorch_lightning import seed_everything

sys.path.append("./stable_diffusion")

from ldm.modules.diffusionmodules.openaimodel import (UNetModel)
from ldm.models.diffusion.ddpm_edit import (LatentDiffusion)
from ldm.util import instantiate_from_config
from ldm.modules.diffusionmodules.util import (
    timestep_embedding, )

sys.path.append("./")


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
            k: vae_sd[k[len("first_stage_model."):]]
            if k.startswith("first_stage_model.") else v
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


class LatentDiffusion_C(LatentDiffusion):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_depth(self, depth):
        unet = self.model.diffusion_model
        unet.depth = depth
        unet.encoder_skip = list(range(unet.depth + 1, 12))
        unet.decoder_skip = list(range(0, 11 - unet.depth))


class UNetModel_C(UNetModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth = 12
        self.encoder_skip = []
        self.decoder_skip = []
        self.dimension_spec = {
            "D0": {
                "skip": 320,
                "input": 320,
                "spatial": 1
            },
            "D1": {
                "skip": 320,
                "input": 320,
                "spatial": 1
            },
            "D2": {
                "skip": 320,
                "input": 640,
                "spatial": 1
            },
            "D3": {
                "skip": 320,
                "input": 640,
                "spatial": 2
            },
            "D4": {
                "skip": 640,
                "input": 640,
                "spatial": 2
            },
            "D5": {
                "skip": 640,
                "input": 1280,
                "spatial": 2
            },
            "D6": {
                "skip": 640,
                "input": 1280,
                "spatial": 4
            },
            "D7": {
                "skip": 1280,
                "input": 1280,
                "spatial": 4
            },
            "D8": {
                "skip": 1280,
                "input": 1280,
                "spatial": 4
            },
            "D9": {
                "skip": 1280,
                "input": 1280,
                "spatial": 8
            },
            "D10": {
                "skip": 1280,
                "input": 1280,
                "spatial": 8
            },
            "D11": {
                "skip": 1280,
                "input": 1280,
                "spatial": 8
            }
        }

    def is_skip(self):
        return self.depth < 12

    def forward(self,
                x,
                timesteps=None,
                context=None,
                struct_cond=None,
                y=None,
                **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps,
                                   self.model_channels,
                                   repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0], )
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for i, module in enumerate(self.input_blocks):
            if self.is_skip() and i in self.encoder_skip:
                continue
            h = module(h, emb, context)
            hs.append(h)

        if self.is_skip():
            h = torch.zeros(h.shape[0],
                            self.dimension_spec[f"D{self.depth}"]["input"],
                            h.shape[2], h.shape[3]).to(h.device)
        else:
            h = self.middle_block(h, emb, context)

        for i, module in enumerate(self.output_blocks):
            if self.is_skip() and i in self.decoder_skip:
                continue
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)


@torch.no_grad()
def main():
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--depth", default=13, type=int)
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.1, type=float)
    parser.add_argument("--ckpt",
                        default="ckpts/instruct-pix2pix-00-22000.ckpt")
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--input", default="imgs/0030.png")
    parser.add_argument("--prompt", default="make his hair red")
    args = parser.parse_args()

    sys.path.append(os.getcwd())
    cfn = Path(os.path.basename(__file__)).stem
    config = OmegaConf.load("configs/depth-skip.yaml")

    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.set_depth(args.depth - 1)
    model.eval().cuda()

    dir_log = join("logs", cfn)
    os.makedirs(dir_log, exist_ok=True)

    null_token = model.get_learned_conditioning([""])

    path_input = args.input
    prompt = args.prompt

    seed = 23
    eta = 1.0
    input_image = Image.open(path_input).convert("RGB")

    with torch.no_grad(), autocast("cuda"), model.ema_scope():
        cond = {}
        cond["c_crossattn"] = [model.get_learned_conditioning([prompt])]
        input_image.save(join("logs", cfn, "input.png"))
        input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
        input_image = rearrange(input_image,
                                "h w c -> 1 c h w").to(model.device)
        cond["c_concat"] = [model.encode_first_stage(input_image).mode()]

        uncond = {}
        uncond["c_crossattn"] = [null_token]
        uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": args.cfg_text,
            "image_cfg_scale": args.cfg_image,
        }
        batch_size = 1

        total = 30
        timesteps = torch.linspace(0, 999, total).to(torch.int32).flip(0)

        torch.manual_seed(seed)
        x_t = torch.randn(batch_size, 4, 64, 64).cuda()
        alphas = model.alphas_cumprod
        sqrt_alphas = model.sqrt_alphas_cumprod
        sqrt_betas = model.sqrt_one_minus_alphas_cumprod

        cnt = len(timesteps)
        for idx in tqdm(range(cnt)):
            t = timesteps[idx]
            ts = torch.full((batch_size, ), t, dtype=torch.long).cuda()

            cfg_z = einops.repeat(x_t, "1 ... -> n ...", n=3)
            cfg_ts = einops.repeat(ts, "1 ... -> n ...", n=3)
            cfg_cond = {
                "c_crossattn": [
                    torch.cat([
                        cond["c_crossattn"][0], uncond["c_crossattn"][0],
                        uncond["c_crossattn"][0]
                    ])
                ],
                "c_concat": [
                    torch.cat([
                        cond["c_concat"][0], cond["c_concat"][0],
                        uncond["c_concat"][0]
                    ])
                ],
            }
            out_cond, out_img_cond, out_uncond = model.apply_model(
                cfg_z, cfg_ts, cond=cfg_cond).chunk(3)
            e_t = out_uncond + args.cfg_text * (
                out_cond - out_img_cond) + args.cfg_image * (out_img_cond -
                                                             out_uncond)

            x_0 = 1 / sqrt_alphas[t] * (x_t - sqrt_betas[t] * e_t)

            if idx == cnt - 1:
                x = model.decode_first_stage(x_0)
                x = x.add(1).mul(0.5).clamp(0, 1)
                img_out = ToPILImage()(x[0])
                img_out.save(join("logs", cfn,
                                  f"output_d{args.depth:02d}.png"))
                break
            tm1 = timesteps[idx + 1]
            sigma = eta * np.sqrt(
                (1 - alphas[tm1].item()) / (1 - alphas[t].item()) *
                (1 - alphas[t].item() / alphas[tm1].item()))
            x_t = sqrt_alphas[tm1] * x_0 + torch.sqrt(
                1 - alphas[tm1] -
                sigma**2) * e_t + sigma * torch.randn_like(x_t)


if __name__ == "__main__":
    main()
