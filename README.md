# EC601 Term Project

Jake Hellman  
jhellman@bu.edu  


Simon Gilbert  
simonsai@bu.edu  


Shashank Manjunath  
manjuns@bu.edu  


## Introduction

## Setting Up The Environment

To create the environment within which to run this project, we use [Anaconda](https://docs.anaconda.com/anaconda/). Once
you have installed Anaconda3, create the environment using the command:

```
conda env create -f environment.yml
```

## manjuns EC601 Project 3

*This portion of the project based on the work of [GitHub user
rekalantar](https://github.com/rekalantar/CT_lung_3D_segmentation)*

From rekalantar:  
*In order to detect lung lesions in CT 3D volumes, the lung needs to be segmented from the images. This project inspired
by the Kaggle Data Science Bowl 2017, aimed to automate 3D lung segmentation from the CT scans using a 3D U-Net model.
These allow calculation of paramterers such as the lung volume and Percentile Density (PD) from the CT scans. The
dataset provides 2D and 3D images along with the masks provided by radiologists. The code was written for use in Google
Colab and the dataset can be directly downloaded from the Kaggle database with personal API tokens.*

We plan to modify the code to apply the model trained by rekalantar to the data from the Kaggle Pulmonary Fibrosis
competition.

