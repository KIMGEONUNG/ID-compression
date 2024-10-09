import sys
import argparse

sys.path.append('.')
from share import *
import os
from itertools import product
from PIL import Image
import config
import cv2
import einops
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import os
from pathlib import Path

###############################################################################
import torch as th
from cldm.cldm import ControlledUnetModel
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.modules.diffusionmodules.util import (
    timestep_embedding, )


class ControlledUnetModel_DS(ControlledUnetModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.depth = 12
        self.encoder_skip = list(range(self.depth + 1, 12))
        self.decoder_skip = list(range(0, 11 - self.depth))

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

    def set_depth(self, depth):
        self.depth = depth
        self.encoder_skip = list(range(self.depth + 1, 12))
        self.decoder_skip = list(range(0, 11 - self.depth))

    def forward(self,
                x,
                timesteps=None,
                context=None,
                control=None,
                only_mid_control=False,
                **kwargs):
        hs = []
        t_emb = timestep_embedding(timesteps,
                                   self.model_channels,
                                   repeat_only=False)
        emb = self.time_embed(t_emb)
        h = x.type(self.dtype)
        for i, module in enumerate(self.input_blocks):
            if self.is_skip() and i in self.encoder_skip:
                control.pop()
                continue
            h = module(h, emb, context)
            hs.append(h)

        if self.is_skip():
            h = th.zeros(h.shape[0],
                         self.dimension_spec[f"D{self.depth}"]["input"],
                         h.shape[2], h.shape[3]).to(h.device)
            control.pop()
        else:
            h = self.middle_block(h, emb, context)
            if control is not None:
                h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if self.is_skip() and i in self.decoder_skip:
                continue
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


###############################################################################


@torch.no_grad()
def process(input_image, prompt, a_prompt, n_prompt, num_samples,
            image_resolution, ddim_steps, guess_mode, strength, scale, seed,
            eta, low_threshold, high_threshold, depth):

    model.model.diffusion_model.set_depth(depth)
    img = resize_image(HWC3(input_image), image_resolution)
    H, W, C = img.shape

    detected_map = apply_canny(img, low_threshold, high_threshold)
    detected_map = HWC3(detected_map)

    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    cond = {
        "c_concat": [control],
        "c_crossattn": [
            model.get_learned_conditioning([prompt + ', ' + a_prompt] *
                                           num_samples)
        ]
    }
    un_cond = {
        "c_concat": None if guess_mode else [control],
        "c_crossattn":
        [model.get_learned_conditioning([n_prompt] * num_samples)]
    }
    shape = (4, H // 8, W // 8)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=True)

    model.control_scales = [
        strength * (0.825**float(12 - i)) for i in range(13)
    ] if guess_mode else (
        [strength] * 13
    )  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(
        ddim_steps,
        num_samples,
        shape,
        cond,
        verbose=False,
        eta=eta,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=un_cond)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 +
                 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

    results = [x_samples[i] for i in range(num_samples)]
    return [255 - detected_map] + results


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--depth', type=int, default=9)
    return p.parse_args()


if __name__ == "__main__":

    args = parse()

    apply_canny = CannyDetector()

    model = create_model('./configs/depthskip.yaml').cpu()
    model.load_state_dict(
        load_state_dict('./models/control_sd15_canny.pth', location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    input_image = np.array(Image.open('test_imgs/bird.png'))
    scale = 9
    low_threshold = 100
    high_threshold = 200

    name = Path(os.path.basename(__file__)).stem
    strength = 1

    depth = args.depth - 1
    eta = 0.0
    seed = 2

    dir = f'logs/{name}'
    os.makedirs(dir, exist_ok=True)
    a = process(
        input_image=input_image,
        prompt="bird",
        a_prompt="best quality, extremely detailed",
        n_prompt=
        "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
        num_samples=1,
        image_resolution=512,
        guess_mode=False,
        ddim_steps=20,
        strength=strength,
        scale=scale,
        seed=seed,
        eta=eta,
        depth=depth,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    img = Image.fromarray(a[1])
    img.save(f'{dir}/d{depth:02d}_{seed:03d}.png')
