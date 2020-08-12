# Tree species classification from from airborne LiDAR and hyperspectral data using 3D convolutional neural networks

## Table of Contents

* [About](#about)
* [Getting started](#about)
  * [Installation](#installation)
* [Data](#data)
* [Codes](#codes)
* [Workflow](#workflow)
  * [Preprocessing](#preprocessing)
  * [Individual tree detection and matching field data to detected tree crowns](#individual-tree-detection-and-matching-field-data-to-detected-tree-crowns)
  * [Training and validation data generation](#training-and-validation-data-generation)
  * [Model training](#model-training)
    * [Reference methods](#reference-methods)
    * [3D CNNs](#3d-cnns)
  * [Inference and interpretation](#inference-and-interpretation)
* [Authors](#authors)

## About

This is a code repository for our paper **Tree species classification from airborne hyperspectral and LiDAR data using 3D convolutional neural networks**, submitted to Remote Sensing of Environment for publication.

> During the last two decades, forest monitoring and inventory systems have moved from field surveys to remote sensing-based methods. These methods tend to focus on economically significant components of forests, thus leaving out many factors vital for forest biodiversity, such as the occurrence of species with low economical but high ecological values. Airborne hyperspectral imagery has shown significant potential for tree species classification. However, the most common analysis methods, such as random forest and support vector machines, fail to utilize spatial information present in the images, only relying on spectral features.
>
> We compared the performance of three-dimensional convolutional neural networks (3D-CNNs) against support vector machine, random forest, gradient boosting machine and feedforward neural network in individual tree species classification from hyperspectral data with high spatial and spectral resolution. We collected hyperspectral and LiDAR data along with extensive ground reference data measurements of tree species from the 83 km² study area located in the southern boreal zone in Finland. A LiDAR-derived canopy height model was used to match ground reference data to aerial imagery. Our research focused on classification of major tree species, Scots pine (*Picea abies* (L.) Karst.), Norway spruce (*Pinus sylvestris*), birch (*Betula* sp., including both *pendula* and *pubescens*), together with a keystone species European aspen (*Populus tremula* L.) that has a sparse and scattered occurrence in boreal forests. Our results showed that 3D-CNNs were able to outperform other methods, with the best performing model achieving macro F1-score of 0.85 and overall accuracy of 0.86. Compared to the reference models, 3D-CNNs were more efficient in distinguishing coniferous species from each other, with a concurrent high accuracy for aspen classification.
>
> Deep neural networks, being a black-box model, hide the information about how they reason their decision. We used both occlusion and saliency maps to gain insight on which input features 3D-CNN puts the most weight. Finally, we used the best-performing 3D-CNN to produce wall-to-wall tree species map for the full study area that can later be used as a ground-truth in other tasks, such as tree species mapping from multispectral satellite images. Improved tree species classification demonstrated by our study can benefit both sustainable forestry and biodiversity conservation.

## Getting started

Project members can access preinstalled conda environment by running `source conda_activate.sh`. This version has fastai2 v0.0.17 and pytorch 1.3.0. This project should work with latest versions (at the time of writing `fastai2==0.0.25` and `pytorch=1.6.0`, but this hasn't been tested yet.

Do not use any CSC modules with this conda environment when running python based scripts and notebooks. 

For R-files, use `module load r-env`

### Installation

Run `conda env create -f environment.yml`, and then install fastai2 either from pip with `pip install fastai2`, or you can use editable install of fastai2 and fastcore:

```bash
git clone https://github.com/fastai/fastcore
cd fastcore
pip install -e ".[dev]"

cd ..
git clone --recurse-submodules https://github.com/fastai/fastai2
cd fastai2
pip install -e ".[dev]"
```

NOTE: Since Pytorch version 1.6, the default save format has changed from Pickle-based to zip-file based. `torch.load` should however work with older versions. This project was done with Pytorch 1.3.0

## Data

Unfortunately, data is not (yet) publically available. Further questions can be sent to authors.

## Workflow

All steps in our work are presented either in Jupyter Notebooks or individual scripts

### Preprocessing

Preprocessing is described in notebook [Preprocessing, segmentation and train data generation](notebooks/Preprocessing%2C%20segmentation%20and%20train%20data%20generation.ipynb).

### Individual tree detection and matching field data to detected tree crowns

Process and steps is described in notebook [Individual tree detection, segmentation and matching to field data](notebooks/Individual%20tree%20detection%2C%20segmentation%20and%20matching%20to%20field%20data.ipynb).

### Training and validation data generation

Process is descried in notebook [Training and validation data generation](notebooks/Training%20and%20validation%20data%20generation.ipynb).

### Model training

#### Reference methods

Training and validation process is presented in notebook [Comparison methods](notebooks/Comparison%20methods.ipynb).

#### 3D CNNs

Todo

### Inference and interpretation

Todo

## Authors

Janne Mäyrä (corresponding author), Sarita Keski-Saari, Sonja Kivinen, Topi Tanhuanpää, Pekka Hurskainen, Peter Kullberg, Laura Poikolainen, Arto Viinikka, Sakari Tuominen, Timo Kumpula, Petteri Vihervaara
