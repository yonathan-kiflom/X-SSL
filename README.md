# X-SSL
Self-supervised xray threat localization


![alt text](https://github.com/yonathan-kiflom/X-SSL/blob/main/assets/Architecture.png)

[![Paper](https://img.shields.io/badge/Paper-Elsevier-orange.svg)](https://www.sciencedirect.com/science/article/pii/S0306457325002808)

**X-SSL** is a self-supervised framework that learns from unlabeled X-ray scans to localize threats, combining zero-shot region proposals, multi-modal clustering, and knowledge distillation.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Model Zoo](#model-zoo)
- [Datasets](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Citation](#citation)
 
## Requirements
- Python >= 3.8
- CUDA >= 11.8 
- PyTorch >= 1.8

## Installation
See Cutler [Installation instructions](https://github.com/facebookresearch/CutLER/blob/cca0a270cf68399efc8fd50df426b6d806e39416/INSTALL.md).



## Model zoo


## Datasets
All datasets used in this paper can be accessed here:
[PIDray](https://github.com/lutao2021/PIDray)
[CLCxray](https://github.com/GreysonPhoenix/CLCXray)

## Training


## Evaluation



## 📄 Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@article{MICHAELX-SSL,
title = {X-SSL: Self-supervised X-ray threat detection with zero-shot and multi-modal learning},
journal = {Information Processing & Management},
volume = {63},
number = {1},
pages = {104339},
year = {2026},
issn = {0306-4573},
doi = {https://doi.org/10.1016/j.ipm.2025.104339},
url = {https://www.sciencedirect.com/science/article/pii/S0306457325002808},
author = {Yonathan Michael and Mohamad Alansari and Abdelfatah Ahmed and Naoufel Werghi and Andreas Henschel}}
