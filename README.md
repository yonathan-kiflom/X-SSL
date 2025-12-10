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
Adopt your choice of ViT based evaluation model for downstream task and replace the backbone with your finetuned model.

## Results
# Detection and Segmentation
<table>
  <thead>
    <tr>
      <th rowspan="3">Model</th>
      <th rowspan="3">Architecture</th>
      <th colspan="4">Easy</th>
      <th colspan="4">Hard</th>
      <th colspan="4">Hidden</th>
      <th colspan="4">CLCxray</th>
    </tr>
    <tr>
      <th colspan="2">Det</th><th colspan="2">Seg</th>
      <th colspan="2">Det</th><th colspan="2">Seg</th>
      <th colspan="2">Det</th><th colspan="2">Seg</th>
      <th colspan="2">Det</th><th colspan="2">Seg</th>
    </tr>
    <tr>
      <th>AP</th><th>AP50</th><th>AP</th><th>AP50</th>
      <th>AP</th><th>AP50</th><th>AP</th><th>AP50</th>
      <th>AP</th><th>AP50</th><th>AP</th><th>AP50</th>
      <th>AP</th><th>AP50</th><th>AP</th><th>AP50</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>FreeSOLO</strong></td><td>ResNet</td>
      <td>51.17</td><td>67.39</td><td>36.52</td><td>65.79</td>
      <td>37.94</td><td>63.78</td><td>25.97</td><td>56.55</td>
      <td>29.62</td><td>43.34</td><td>18.90</td><td>40.36</td>
      <td>--</td><td>--</td><td>--</td><td>--</td>
    </tr>
    <tr>
      <td><strong>FreeSOLO + Ours</strong></td><td>ResNet</td>
      <td><strong>61.03</strong></td><td><strong>76.77</strong></td><td><strong>45.53</strong></td><td><strong>75.60</strong></td>
      <td><strong>48.89</strong></td><td><strong>73.13</strong></td><td><strong>34.42</strong></td><td><strong>67.95</strong></td>
      <td><strong>35.23</strong></td><td><strong>51.55</strong></td><td><strong>21.09</strong></td><td><strong>46.26</strong></td>
      <td>--</td><td>--</td><td>--</td><td>--</td>
    </tr>
    <tr style="background: #f4f7fb;">
      <td><strong>Improvement</strong></td><td>-----</td>
      <td>+9.86</td><td>+9.38</td><td>+9.01</td><td>+9.81</td>
      <td>+10.95</td><td>+9.35</td><td>+8.45</td><td>+11.40</td>
      <td>+5.61</td><td>+8.21</td><td>+2.19</td><td>+5.90</td>
      <td>--</td><td>--</td><td>--</td><td>--</td>
    </tr>
    <tr>
      <td><strong>LOST</strong></td><td>ViTSmall</td>
      <td>25.66</td><td><strong>61.79</strong></td><td>--</td><td>--</td>
      <td>14.88</td><td>47.34</td><td>--</td><td>--</td>
      <td>7.38</td><td>21.01</td><td>--</td><td>--</td>
      <td>33.32</td><td><strong>64.23</strong></td><td>--</td><td>--</td>
    </tr>
    <tr>
      <td><strong>CutLER (pretrained)</strong></td><td>ViTSmall</td>
      <td>20.98</td><td>31.79</td><td>16.45</td><td>30.11</td>
      <td>17.80</td><td>29.22</td><td>13.58</td><td>24.55</td>
      <td>16.88</td><td>24.47</td><td>12.35</td><td>21.97</td>
      <td>27.74</td><td>40.08</td><td>26.33</td><td>39.73</td>
    </tr>
    <tr>
      <td><strong>CutLER + finetuned DINO</strong></td><td>ViTSmall</td>
      <td>20.60</td><td>32.95</td><td>16.47</td><td>30.13</td>
      <td>24.91</td><td>40.15</td><td>16.80</td><td>32.83</td>
      <td>6.63</td><td>13.74</td><td>5.19</td><td>12.50</td>
      <td>29.44</td><td>46.68</td><td>28.47</td><td>46.65</td>
    </tr>
    <tr>
      <td><strong>CutLER + Ours</strong></td><td>ViTSmall</td>
      <td><strong>33.41</strong></td><td>51.15</td><td><strong>30.55</strong></td><td><strong>49.73</strong></td>
      <td><strong>29.49</strong></td><td><strong>49.27</strong></td><td><strong>22.02</strong></td><td><strong>42.60</strong></td>
      <td><strong>17.18</strong></td><td><strong>27.79</strong></td><td><strong>12.77</strong></td><td><strong>25.25</strong></td>
      <td><strong>38.97</strong></td><td>55.13</td><td><strong>38.36</strong></td><td><strong>55.17</strong></td>
    </tr>
    <tr style="background: #f4f7fb;">
      <td><strong>Improvement</strong></td><td>-----</td>
      <td>+7.75</td><td>-10.64</td><td>+14.08</td><td>+19.60</td>
      <td>+4.58</td><td>+1.93</td><td>+5.22</td><td>+9.77</td>
      <td>+0.30</td><td>+3.32</td><td>+0.42</td><td>+3.28</td>
      <td>+5.65</td><td>-9.10</td><td>+9.89</td><td>+8.52</td>
    </tr>
    <tr>
      <td><strong>CutLER + finetuned DINO</strong></td><td>ViTBase</td>
      <td>21.88</td><td>34.77</td><td>17.15</td><td>31.31</td>
      <td>26.56</td><td>42.33</td><td>18.53</td><td>34.98</td>
      <td>6.33</td><td>13.11</td><td>4.90</td><td>12.10</td>
      <td>28.15</td><td>45.38</td><td>26.33</td><td>45.45</td>
    </tr>
    <tr>
      <td><strong>CutLER + Ours</strong></td><td>ViTBase</td>
      <td><strong>33.24</strong></td><td><strong>52.18</strong></td><td><strong>28.08</strong></td><td><strong>48.75</strong></td>
      <td><strong>27.31</strong></td><td><strong>48.14</strong></td><td><strong>21.61</strong></td><td><strong>41.20</strong></td>
      <td><strong>17.64</strong></td><td><strong>29.80</td><td><strong>12.82</strong></td><td><strong>26.86</strong></td>
      <td><strong>36.33</strong></td><td><strong>53.16</strong></td><td><strong>35.96</strong></td><td><strong>53.10</strong></td>
    </tr>
    <tr style="background: #f4f7fb;">
      <td><strong>Improvement</strong></td><td>-----</td>
      <td>+11.36</td><td>+17.41</td><td>+10.93</td><td>+17.44</td>
      <td>+0.75</td><td>+5.81</td><td>+3.08</td><td>+6.22</td>
      <td>+11.31</td><td>+16.69</td><td>+7.92</td><td>+14.76</td>
      <td>+8.18</td><td>+7.78</td><td>+9.63</td><td>+7.65</td>
    </tr>
  </tbody>
</table>



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
