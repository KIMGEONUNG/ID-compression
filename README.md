<div align="center">

<h1>
    ID-Compression:<br> 
    Diffusion Model Compression for Image-to-Image Translation
</h1>

<div>
    <a href='https://kimgeonung.github.io/' target='_blank'>Geonung Kim</a>&emsp;
    <a target='_blank'>Beomsu Kim</a>&emsp;
    <a target='_blank'>Eunhyeok Park</a>&emsp;
    <a href='https://www.scho.pe.kr/' target='_blank'>Sunghyun Cho</a>&emsp;
</div>
<div>
    POSTECH
</div>

<div>
    <strong>ACCV 2024 </strong>
</div>

<div>
    <h4 align="center">
        <a href="https://kimgeonung.github.io/id-compression/" target='_blank'>
        <img src="https://img.shields.io/badge/🐳-Project%20Page-blue">
        </a>
        <a href="https://arxiv.org/abs/2401.17547" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2401.17547-b31b1b.svg">
        </a>
    </h4>
</div>

![teaser](assets/teaser.png) 
---

</div>

## 🔥 Update

- [2025.01.25] Time-step optimization code for StableSR is released
- [2024.10.09] Depth-skip compression code is released
- [2024.10.08] The repository is created.


## 🔧 Install Environment

### Clone Repository

```bash
git clone git@github.com:KIMGEONUNG/ID-compression.git
```

### Conda Environment

We offer three applications using our compression method: InstructPix2Pix for image editing, StableSR for image restoration, and ControlNet for structure-guided image synthesis. Please refer to the original application repository to install the conda environments.

- [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix/tree/main)
- [StableSR](https://github.com/IceClear/StableSR)
- [ControlNet](https://github.com/lllyasviel/ControlNet )

### Checkpoints

The checkpoint paths for each task are as follows. Please refer to the original repository to download the checkpoints.

```bash
# For InstructPix2Pix
InstructPix2Pix/ckpts
└── instruct-pix2pix-00-22000.ckpt

# For StableSR
StableSR/ckpts
├── stablesr_000117.ckpt
└── vqgan_cfw_00011.ckpt

# For ControlNet
ControlNet/models
└── control_sd15_canny.pth 
```

## ⌨️  Quick Start

### InstructPix2Pix

```bash
cd InstructPix2Pix
python depth-skip.py --depth 9
```

### StableSR

```bash
cd StableSR
python depth-skip.py --depth 9
```

### ControlNet

```bash
cd ControlNet
python depth-skip.py --depth 9
```

## 📃 Reported Optimized Time-steps

### InstructPix2Pix

| Step | Time-step sequence                                                                                   |
|------|------------------------------------------------------------------------------------------------------|
| 5    | [999, 916, 814, 674, 403]                                                                            |
| 10   | [999, 937, 872, 802, 728, 646, 556, 452, 324, 124]                                                   |
| 15   | [999, 948, 896, 843, 789, 733, 676, 616, 554, 488, 419, 345, 264, 170, 43]                           |
| 20   | [999, 958, 917, 876, 834, 791, 748, 704, 659, 613, 566, 518, 468, 417, 364, 309, 250, 187, 116, 26]  |

### StableSR

| Step | Time-step sequence                                                                            |
|------|-----------------------------------------------------------------------------------------------|
|5	   |[949, 549, 254, 68, 0]                                                                         |
|10	   |[970, 804, 649, 507, 379, 265, 167, 87, 28, 0]                                                 |
|15	   |[970, 861, 756, 658, 564, 476, 394, 317, 247, 184, 129, 81, 42, 13, 0]                         |
|20	   |[981, 908, 837, 768, 701, 636, 572, 511, 451, 394, 339, 287, 237, 190, 147, 107, 71, 40, 14, 0]|

### ControlNet

| Step | Time-step sequence                                                                                   |
|------|------------------------------------------------------------------------------------------------------|
| 5    | [999, 881, 741, 558, 219]                                                                            |
| 10   | [999, 930, 858, 782, 701, 614, 518, 410, 279, 84]                                                    |
| 15   | [999, 948, 896, 843, 789, 733, 676, 616, 554, 488, 419, 345, 264, 170, 43]                           |
| 20   | [999, 990, 969, 953, 895, 878, 773, 617, 511, 436, 328, 247, 124, 61, 41, 29, 27, 17, 17, 0]         |


## ☕️ Acknowledgment

- We borrowed the readme format from [Upscale-A-Video](https://github.com/sczhou/Upscale-A-Video) 
