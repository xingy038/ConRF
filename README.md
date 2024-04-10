# ConRF: Zero-shot Stylization of 3D Scenes with Conditioned Radiation Fields
## [Project page](https://xingy038.github.io/ConRF/) |  [Paper](https://arxiv.org/pdf/2402.01950)

This repository contains a pytorch implementation for the paper: [ConRF: Zero-shot Stylization of 3D Scenes with Conditioned Radiation Fields](https://arxiv.org/pdf/2402.01950).  


## Installation
> Tested on Ubuntu 20.04 + Pytorch 1.12.1

Install environment:
```
conda create -n ConRF python=3.9
conda activate ConRF
pip install torch torchvision
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard ftfy regex
$ pip install git+https://github.com/openai/CLIP.git 
```

## Datasets
Please put the datasets in `./data`. You can put the datasets elsewhere if you modify the corresponding paths in the configs.

### 3D scene datasets
* [nerf_synthetic](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) 
* [llff](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
### Style image dataset
* [WikiArt](https://www.kaggle.com/datasets/ipythonx/wikiart-gangogh-creating-art-gan)

## Quick Start
The checkpoint will release soon.

Then modify the following attributes in `scripts/test_style.sh`:
* `--config`: choose `configs/llff_style.txt` or `configs/nerf_synthetic_style.txt` according to which type of dataset is being used
* `--datadir`: dataset's path
* `--ckpt`: checkpoint's path
* `--style_img`: reference style image's path
* `--text`: reference text style


To generate stylized novel views:
```
bash scripts/test_style.sh [GPU ID]
```
The rendered stylized images can then be found in the directory under the checkpoint's path.

## Training
> Current settings in `configs` are tested on one NVIDIA RTX A100 Graphics Card with 80G memory. To reduce memory consumption, you can set `batch_size`, `chunk_size` or `patch_size` to a smaller number.

We follow StyleRF for the following 3 steps of training:
### 1. Train original TensoRF
This step is for reconstructing the density field, which contains more precise geometry details compared to mesh-based methods. You can skip this step by directly downloading pre-trained checkpoints provided by [TensoRF checkpoints](https://1drv.ms/u/s!Ard0t_p4QWIMgQ2qSEAs7MUk8hVw?e=dc6hBm).

The configs are stored in `configs/llff.txt` and `configs/nerf_synthetic.txt`. For the details of the settings, please also refer to [TensoRF](https://github.com/apchenstu/TensoRF). The checkpoints are stored in `./log` by default.

You can train the original TensoRF by:
```
bash script/train.sh [GPU ID]
```

### 2. Feature grid training stage
This step is for reconstructing the 3D gird containing the VGG features.

The configs are stored in `configs/llff_feature.txt` and `configs/nerf_synthetic_feature.txt`, in which `ckpt` specifies the checkpoints trained in the **first** step. The checkpoints are stored in `./log_feature` by default.

Then run:
```
bash script/train_feature.sh [GPU ID]
```


### 3. Stylization training stage 
This step is for training the style transfer modules.

The configs are stored in `configs/llff_style.txt` and `configs/nerf_synthetic_style.txt`, in which `ckpt` specifies the checkpoints trained in the **second** step. The checkpoints are stored in `./log_style` by default.

Then run:
```
bash script/train_style.sh [GPU ID]
```

There may be some unknown errors due to cleaning up the code, please let me know. If you have any concern with this paper or implementation, welcome to open an issue or email me at xingyu.miao@durham.ac.uk.

## Acknowledgments
This repo is heavily based on the [TensoRF](https://github.com/apchenstu/TensoRF) and [StyleRF](https://kunhao-liu.github.io/StyleRF/). Thank them for sharing their amazing work!

## Citation
If you find our code or paper helps, please consider citing:
```
@misc{miao2024conrf,
      title={ConRF: Zero-shot Stylization of 3D Scenes with Conditioned Radiation Fields}, 
      author={Xingyu Miao and Yang Bai and Haoran Duan and Fan Wan and Yawen Huang and Yang Long and Yefeng Zheng},
      year={2024},
      eprint={2402.01950},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

