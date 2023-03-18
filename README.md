# AETNet

This is the official repository for  [“You Only Train Once: Learning a General Anomaly Enhancement Network with Random Masks for Hyperspectral Anomaly Detection”](https://ieeexplore.ieee.org/document/10073635). 

## Abstract

In this paper, we introduce a new approach to address the challenge of generalization in hyperspectral anomaly detection (AD). Our method eliminates the need for adjusting parameters or retraining on new test scenes as required by most existing methods. Employing an image-level training paradigm, we achieve a general anomaly enhancement network for hyperspectral AD that only needs to be trained once. Trained on a set of anomaly-free hyperspectral images with random masks, our network can learn the spatial context characteristics between anomalies and background in an unsupervised way. Additionally, a plug-and-play model selection module is proposed to search for a spatial-spectral transform domain that is more suitable for AD task than the original data. To establish a unified benchmark to comprehensive evaluate our method and existing methods, we develop a large-scale hyperspectral AD dataset (HAD100) that includes 100 real test scenes with diverse anomaly targets. In comparison experiments, we combine our network with a parameter-free detector, and achieve the optimal balance between detection accuracy and inference speed among state-of-the-art AD methods. Experimental results also show that our method still achieves competitive performance when the training and test set are captured by different sensor devices.



## Getting Started


###Requirements

- torch
- kornia
- sklearn
- timm
- scikit-image

### Data Preparation

1. Download and unzip  [HAD100dataset]( https://zhaoxuli123.github.io/HAD100/).  
2. Put the generated folder into ./data

### Train 
```shell
python train.py
```

### Test
```shell
python test.py 
```


## Citation

If the work or the code is helpful, please cite the paper:

```
@ARTICLE{10073635,
  author={Li, Zhaoxu and Wang, Yingqian and Xiao, Chao and Ling, Qiang and Lin, Zaiping and An, Wei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={You Only Train Once: Learning a General Anomaly Enhancement Network with Random Masks for Hyperspectral Anomaly Detection}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TGRS.2023.3258067}}
```

## Contact
For further questions or details, please directly reach out to lizhaoxu@nudt.edu.cn