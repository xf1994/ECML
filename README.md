#ECML: An Ensemble Cascade Metric Learning Mechanism towards Face Verification

## Introduction
This is the code for the paper, **"ECML: An Ensemble Cascade Metric Learning Mechanism towards Face Verification"**. 

In this paper, we propose a novel metric learning mechanism towards Face Verification called ECML. In addition, a robust
Mahalanobis metric learning method (RMML) with closed-form solution is additionally proposed.

# About our code 
## preparision
Matlab is needed for running for this code. Besides, face feature used in this paper can be downloaded from https://pan.baidu.com/s/1RYK3uITiHpIctJmhsaHnLQ (access code: x3hd), where f.mat denotes the CNN features and encoding.mat denotes the FV features.

## configuration
To config the code, simply modify the 'opt' structutre in ECRMML.m, where you can choose the layers and core linear metric learning method for ECML.
