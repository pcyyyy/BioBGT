#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 17:19:28 2024

@author: sayeon
"""
from nilearn.maskers import NiftiSpheresMasker
import numpy as np
from nilearn import datasets, connectome
from sklearn.covariance import GraphicalLassoCV
from nilearn import plotting
from skimage import io,data
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from nilearn.interfaces.fmriprep import load_confounds
from scipy.io import loadmat
from nilearn.connectome import ConnectivityMeasure
import os
import pandas as pd

data_path ='./atten_sore_new.npy'


score = np.load(data_path, allow_pickle=True)
#print(np.max(score))
#print(np.min(score))
score[score < 0] = 0
score = (score - np.min(score)) / (np.max(score) - np.min(score))
mean_attention_scores = np.mean(score, axis=2)
#print(np.max(mean_attention_scores))
#print(np.min(mean_attention_scores))
# for i in range(score.shape[2]):
# Temp = score[:,:,i]
# Temp[Temp < 0] = 0


#mean_attention_scores[mean_attention_scores < 0] = 0
mean_attention_scores = mean_attention_scores[15:, 15:]
#mean_attention_scores = np.delete(mean_attention_scores, np.s_[50:61], axis=0)

plt.figure(figsize=(10, 10))
colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
cmap = LinearSegmentedColormap.from_list('my_cmap', colors)
plt.imshow(mean_attention_scores, cmap=cmap, interpolation='nearest', vmin=0.14, vmax=0.27)
plt.colorbar()
# plt.xticks(range(len(label)), label, rotation=90, fontsize=6)
# plt.yticks(range(len(new_labels)), new_labels, fontsize=5)

plt.tight_layout()
plt.show()