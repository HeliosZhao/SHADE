# Style-Hallucinated Dual Consistency Learning for Domain Generalized Semantic Segmentation (ECCV 2022)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<img alt="PyTorch" height="20" src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white" />

This repository contains the official implementation of our paper in:

> **[Style-Hallucinated Dual Consistency Learning for Domain Generalized Semantic Segmentation](https://arxiv.org/pdf/2204.02548.pdf)**
> 
> [Yuyang Zhao](http://yuyangzhao.com), [Zhun Zhong](http://zhunzhong.site), [Na Zhao](https://na-z.github.io/), [Nicu Sebe](https://disi.unitn.it/~sebe/), [Gim Hee Lee](https://www.comp.nus.edu.sg/~leegh/)


> **Abstract:**
> In this paper, we study the task of synthetic-to-real domain generalized semantic segmentation, which aims to learn a model that is robust to unseen real-world scenes using only synthetic data. The large domain shift between synthetic and real-world data, including the limited source environmental variations and the large distribution gap between synthetic and real-world data, significantly hinders the model performance on unseen real-world scenes. In this work, we propose the Style-HAllucinated Dual consistEncy learning (SHADE) framework to handle such domain shift. Specifically, SHADE is constructed based on two consistency constraints, Style Consistency (SC) and Retrospection Consistency (RC). SC enriches the source situations and encourages the model to learn consistent representation across style-diversified samples. RC leverages real-world knowledge to prevent the model from overfitting to synthetic data and thus largely keeps the representation consistent between the synthetic and real-world models. Furthermore, we present a novel style hallucination module (SHM) to generate style-diversified samples that are essential to consistency learning. SHM selects basis styles from the source distribution, enabling the model to dynamically generate diverse and realistic samples during training. Experiments show that our SHADE yields significant improvement and outperforms state-of-the-art methods by 5.05% and 8.35% on the average mIoU of three real-world datasets on single- and multi-source settings, respectively.
<br>
<p align="center">
  <img src="assets/shade_intro.png" />
</p>


<!-- ## Pytorch Implementation -->
### Installation
This paper is trained and evaluated on two NVIDIA RTX 3090 GPUs.

Clone this repository and install the following packages. 
```
conda create -n shade python=3.7
conda activate shade
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
pip install -r requirements.txt
imageio_download_bin freeimage
```

### Data Preparation
We trained our model with the source domain [GTAV](https://download.visinf.tu-darmstadt.de/data/from_games/) and [Synthia](https://synthia-dataset.net/downloads/) ([SYNTHIA-RAND-CITYSCAPES](http://synthia-dataset.net/download/808/)). Then we evaluated the model on [Cityscapes](https://www.cityscapes-dataset.com/), [BDD-100K](https://bdd-data.berkeley.edu/), and [Mapillary Vistas](https://www.mapillary.com/dataset/vistas?pKey=2ix3yvnjy9fwqdzwum3t9g&lat=20&lng=0&z=1.5).

Following [RobustNet](https://github.com/shachoi/RobustNet), we adopt Class uniform sampling proposed in [this paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Improving_Semantic_Segmentation_via_Video_Propagation_and_Label_Relaxation_CVPR_2019_paper.pdf) to handle class imbalance problems. 


1. We used [GTAV_Split](https://download.visinf.tu-darmstadt.de/data/from_games/code/read_mapping.zip) to split GTAV dataset into train/val/test set.

```
GTAV
 └ images
 └ labels
```

2. We split [Synthia dataset](http://synthia-dataset.net/download/808/) into train/val set following the [RobustNet](https://github.com/shachoi/RobustNet).
```
synthia
 └ RGB
 └ GT
   └ COLOR
   └ LABELS
```


3. For [Cityscapes](https://www.cityscapes-dataset.com/), download "leftImg8bit_trainvaltest.zip" and "gtFine_trainvaltest.zip". Unzip the files and make the directory structures as follows.
```
CityScapes
  └ leftImg8bit
    └ train
    └ val
    └ test
  └ gtFine
    └ train
    └ val
    └ test
```
4. For [BDD-100K](https://bdd-data.berkeley.edu/), download "10K Images" and "Segmentation". Unzip the files and make the directory structures as follows.
```
bdd100k
 └ images/10k
   └ train
   └ val
   └ test
 └ labels/semseg/masks
   └ train
   └ val
```
5. For [Mapillary Vistas](https://www.mapillary.com/dataset/vistas?pKey=2ix3yvnjy9fwqdzwum3t9g&lat=20&lng=0&z=1.5), download the full dataset. Unzip the files and make the directory structures as follows.
```
mapillary
 └ training
   └ images
   └ labels
 └ validation
   └ images
   └ labels
 └ test
   └ images
   └ labels
```

### Run
You should modify the path in **"<path_to_SHADE>/config.py"** according to your dataset path.
```
#Cityscapes Dir Location
__C.DATASET.CITYSCAPES_DIR = <YOUR_CITYSCAPES_PATH>
#Mapillary Dataset Dir Location
__C.DATASET.MAPILLARY_DIR = <YOUR_MAPILLARY_PATH>
#GTAV Dataset Dir Location
__C.DATASET.GTAV_DIR = <YOUR_GTAV_PATH>
#BDD-100K Dataset Dir Location
__C.DATASET.BDD_DIR = <YOUR_BDD_PATH>
#Synthia Dataset Dir Location
__C.DATASET.SYNTHIA_DIR = <YOUR_SYNTHIA_PATH>
```
#### Train
You can train SHADE with following commands.
```
CUDA_VISIBLE_DEVICES=0,1 sh scripts/train_r50os16_gtav.sh # Train: GTAV, Test: Cityscapes, BDD100K, Mapillary / ResNet50, SHADE
CUDA_VISIBLE_DEVICES=0,1 sh scripts/train_r101os16_gtav.sh # Train: GTAV, Test: Cityscapes, BDD100K, Mapillary / ResNet101, SHADE
CUDA_VISIBLE_DEVICES=0,1 sh scripts/train_r50os16_multi.sh # Train: GTAV, Synthia Test: Cityscapes, BDD100K, Mapillary / ResNet50, SHADE
```
#### Test
You can test SHADE with following commands. You can download models evaluated in our paper at [Release](https://github.com/HeliosZhao/SHADE/releases/tag/v1.0.0).
```
CUDA_VISIBLE_DEVICES=0 sh scripts/valid_r50os16.sh # Test: Cityscapes, BDD100K, Mapillary / ResNet50
CUDA_VISIBLE_DEVICES=0 sh scripts/valid_r101os16.sh # Test: Cityscapes, BDD100K, Mapillary / ResNet101
```

### Acknowledgments
Our pytorch implementation is heavily derived from [RobustNet](https://github.com/shachoi/RobustNet) (CVPR 2021). If you use this code in your research, please also acknowledge their work. [[link to license](https://github.com/shachoi/RobustNet/blob/main/LICENSE)]

### Citation
```
@inproceedings{zhao2022shade,
  title={Style-Hallucinated Dual Consistency Learning for Domain Generalized Semantic Segmentation},
  author={Zhao, Yuyang and Zhong, Zhun and Zhao, Na and Sebe, Nicu and Lee, Gim Hee},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}}
```