# [CVPR 2023] Official Implementation of VideoMAE V2

![flowchart](misc/VideoMAEv2_flowchart.png)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/videomae-v2-scaling-video-masked-autoencoders/spatio-temporal-action-localization-on-ava)](https://paperswithcode.com/sota/spatio-temporal-action-localization-on-ava?p=videomae-v2-scaling-video-masked-autoencoders)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/videomae-v2-scaling-video-masked-autoencoders/action-recognition-on-ava-v2-2)](https://paperswithcode.com/sota/action-recognition-on-ava-v2-2?p=videomae-v2-scaling-video-masked-autoencoders)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/videomae-v2-scaling-video-masked-autoencoders/temporal-action-localization-on-fineaction)](https://paperswithcode.com/sota/temporal-action-localization-on-fineaction?p=videomae-v2-scaling-video-masked-autoencoders)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/videomae-v2-scaling-video-masked-autoencoders/action-recognition-in-videos-on-hmdb-51)](https://paperswithcode.com/sota/action-recognition-in-videos-on-hmdb-51?p=videomae-v2-scaling-video-masked-autoencoders)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/videomae-v2-scaling-video-masked-autoencoders/temporal-action-localization-on-thumos14)](https://paperswithcode.com/sota/temporal-action-localization-on-thumos14?p=videomae-v2-scaling-video-masked-autoencoders)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/videomae-v2-scaling-video-masked-autoencoders/action-recognition-in-videos-on-ucf101)](https://paperswithcode.com/sota/action-recognition-in-videos-on-ucf101?p=videomae-v2-scaling-video-masked-autoencoders)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/videomae-v2-scaling-video-masked-autoencoders/action-recognition-in-videos-on-something-1)](https://paperswithcode.com/sota/action-recognition-in-videos-on-something-1?p=videomae-v2-scaling-video-masked-autoencoders)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/videomae-v2-scaling-video-masked-autoencoders/action-recognition-in-videos-on-something)](https://paperswithcode.com/sota/action-recognition-in-videos-on-something?p=videomae-v2-scaling-video-masked-autoencoders)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/videomae-v2-scaling-video-masked-autoencoders/action-classification-on-kinetics-400)](https://paperswithcode.com/sota/action-classification-on-kinetics-400?p=videomae-v2-scaling-video-masked-autoencoders)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/videomae-v2-scaling-video-masked-autoencoders/action-classification-on-kinetics-600)](https://paperswithcode.com/sota/action-classification-on-kinetics-600?p=videomae-v2-scaling-video-masked-autoencoders)<br>

> [**VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking**](https://arxiv.org/abs/2303.16727)<br>
> [Limin Wang](http://wanglimin.github.io/), [Bingkun Huang](https://github.com/congee524), [Zhiyu Zhao](https://github.com/JerryFlymi), [Zhan Tong](https://scholar.google.com/citations?user=6FsgWBMAAAAJ), [Yinan He](https://dblp.org/pid/93/7763.html), [Yi Wang](https://scholar.google.com.hk/citations?hl=zh-CN&user=Xm2M8UwAAAAJ), [Yali Wang](https://scholar.google.com/citations?user=hD948dkAAAAJ), and [Yu Qiao](https://scholar.google.com/citations?user=gFtI-8QAAAAJ&hl)<br>
> Nanjing University, Shanghai AI Lab, CAS<br>

## News
**[2023.04.19]** ViT-giant model weights have been released! You can get the download links from [MODEL_ZOO.md](docs/MODEL_ZOO.md).<br>
**[2023.04.18]** Code and the distilled models (vit-s & vit-b) have been released!<br>
**[2023.04.03]** ~~Code and models will be released soon.~~<br>


## Model Zoo

We now provide the model weights in [MODEL_ZOO.md](docs/MODEL_ZOO.md).

## Installation

Please follow the instructions in [INSTALL.md](docs/INSTALL.md).

## Data Preparation

Please follow the instructions in [DATASET.md](docs/DATASET.md) for data preparation.

## Pre-training

The pre-training instruction is in [PRETRAIN.md](docs/PRETRAIN.md).

## Fine-tuning

The fine-tuning instruction is in [FINETUNE.md](docs/FINETUNE.md).

## Citation

If you find this repository useful, please use the following BibTeX entry for citation.

```latex
@misc{wang2023videomae,
      title={VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking},
      author={Limin Wang and Bingkun Huang and Zhiyu Zhao and Zhan Tong and Yinan He and Yi Wang and Yali Wang and Yu Qiao},
      year={2023},
      eprint={2303.16727},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
