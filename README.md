# NeRP: Implicit Neural Representation Learning With Prior Embedding for Sparsely Sampled Image Reconstruction


## Contents

- [Overview](#overview)
- [Installation Guide](#installation-guide)
- [Instructions for Running Code](#instructions-for-running-code)
- [License](#license)
- [Citation](#citation)

# 1. Overview

This repository provides the PyTorch code for our TNNLS 2022 papar [NeRP: Implicit Neural Representation Learning With Prior Embedding for Sparsely Sampled Image Reconstruction](https://ieeexplore.ieee.org/document/9788018).

by Liyue Shen, John Pauly, Lei Xing

We propose an implicit <B>Ne</B>ural <B>R</B>epresentation learning methodology with <B>P</B>rior embedding (<B>NeRP</B>) to reconstruct a computational image from sparsely sampled measurements. The method differs fundamentally from previous deep learning-based image reconstruction approaches in that NeRP exploits the internal information in an image prior, and the physics of the sparsely sampled measurements to produce a representation of the unknown subject. No large-scale data is required to train the NeRP except for a prior image and sparsely sampled measurements. We demonstrate that NeRP is a general methodology that generalizes to different imaging modalities including 2D / 3D CT and MRI. 

<p align="center">
  <img src="https://github.com/liyues/NeRP/blob/main/img/Figure_1.svg" width="1200" height="600">
</p>

# 2. Installation Guide

Before running this package, users should have `Python`, `PyTorch`, and several python packages installed (`numpy`, `skimage`, `yaml`, `opencv`, `odl`) .


## Package Versions

This code functions with following dependency packages. The versions of software are, specifically:
```
python: 3.7.4
pytorch: 1.4.1
numpy: 1.19.4
skimage: 0.17.2
yaml: 0.1.7
opencv: 3.4.2
odl: 1.0.0.dev0
```


## Package Installment

Users should install all the required packages shown above prior to running the algorithm. Most packages can be installed by running following command in terminal on Linux. To install PyTorch, please refer to their official [website](https://pytorch.org). To install ODL, please refer to their official [website](https://github.com/odlgroup/odl).

```
pip install package-name
```



# 3. Instructions for Running Code


## 2D CT Reconstruction Experiment

The experiments of 2D CT image reconstruction use the 2D parallel-beam geometry.

### Step 1: Prior embedding

Represent 2D image by implicit network network. The prior image ([pancs_4dct_phase1.npz](./data/ct_data/pancs_4dct_phase1.npz): phase-1 image of a 10-phase 4D pancreas CT data) is provided under [data/ct_data](./data/ct_data) folder.

```
python train_image_regression.py --config configs/image_regression.yaml
```

### Step 2: Network training

Reconstruct 2D CT image from sparsely sampled projections. The reconstruction target image ([pancs_4dct_phase6.npz](./data/ct_data/pancs_4dct_phase6.npz): phase-6 image of a 10-phase 4D pancreas CT data) is provided under [data/ct_data](./data/ct_data) folder.

With prior embedding:
```
python train_ct_recon.py --config configs/ct_recon.yaml --pretrain
```

Without prior embedding:
```
python train_ct_recon.py --config configs/ct_recon.yaml
```

## 3D CT Reconstruction Experiment

The experiments of 3D CT image reconstruction use the 3D cone-beam geometry.

### Step 1: Prior embedding

Represent 3D image by implicit network network. The prior image ([pancs_4dct_phase1.npz](./data/ct_data/pancs_4dct_phase1.npz): phase-1 image of a 10-phase 4D pancreas CT data) is provided under [data/ct_data](./data/ct_data) folder.

```
python train_image_regression_3d.py --config configs/image_regression_3d.yaml
```

### Step 2: Network training

Reconstruct 3D CT image from sparsely sampled projections. The reconstruction target image ([pancs_4dct_phase6.npz](./data/ct_data/pancs_4dct_phase6.npz): phase-6 image of a 10-phase 4D pancreas CT data) is provided under [data/ct_data](./data/ct_data) folder.

With prior embedding:
```
train_ct_recon_3d.py --config configs/ct_recon_3d.yaml --pretrain
```

Without prior embedding:
```
python train_ct_recon_3d.py --config configs/ct_recon_3d.yaml
```

### Step 3: Image inference

Output and save the reconstruted 3D image after training is done at a specified iteration step.

With prior embedding:
```
python test_ct_recon_3d.py --config configs/ct_recon_3d.yaml --pretrain --iter 2000
```

Without prior embedding:
```
python test_ct_recon_3d.py --config configs/ct_recon_3d.yaml --iter 2000
```


# 4. License
A provisional patent application for the reported work has been filed. The codes are copyrighted by Stanford University and are for research only. Correspondence should be addressed to the corresponding author in the paper. Licensing of the reported technique and codes is managed by the Office of Technology Licensing (OTL) of Stanford University.



# 5. Citation
If you find the code are useful, please consider citing the paper.
```
@article{shen2022nerp,
  title={NeRP: implicit neural representation learning with prior embedding for sparsely sampled image reconstruction},
  author={Shen, Liyue and Pauly, John and Xing, Lei},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2022},
  publisher={IEEE}
}
```
