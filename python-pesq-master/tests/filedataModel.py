# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 15:18:39 2022

@author: NeoChen
"""

from pathlib import Path
import scipy.io.wavfile
from scipy import signal

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# read file #
data_dir = Path(__file__).parent.parent / 'audio'
# =============================================================================
ref_path = data_dir / 'speech.wav'
deg_path = data_dir / 'speech_bab_0dB.wav'
# =============================================================================
#ref_path = data_dir / 'vocal_01081.wav'
#deg_path = data_dir / 'mixed_01081_jackhammer.wav'
sample_rate1, ref = scipy.io.wavfile.read(ref_path)
sample_rate2, deg = scipy.io.wavfile.read(deg_path)

f1, t1, Zxx1 = signal.stft(ref, sample_rate1, nperseg=1000)
f2, t2, Zxx2 = signal.stft(deg, sample_rate2, nperseg=1000)

# plot voice data #
# =============================================================================
# plt.scatter(x,y1,c="red")
# plt.scatter(x,y2,c="green")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.title("Scatter Plot of two different datasets")
# plt.show()
# =============================================================================
Zxx3 = Zxx2 - Zxx1
diff_dr = deg - ref
X4, y4 = make_blobs(500, 133, centers=2, random_state=2, cluster_std=1.5)

plt.scatter(deg,ref,c="red")
plt.scatter(deg,diff_dr,c="green")
plt.scatter(X4,X4,c="yellow")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter Plot of two different datasets")
plt.show()

plt.scatter(Zxx2,Zxx1,c="red")
plt.scatter(Zxx2,Zxx3,c="green")
plt.scatter(X4,X4,c="yellow")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter Plot of two different datasets")
plt.show()

# build model #
# =============================================================================
# X, y = make_blobs(150, 2, centers=2, random_state=2, cluster_std=1.5)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu');
# 
# 
# from sklearn.naive_bayes import GaussianNB
# model = GaussianNB()
# model.fit(X, y);
# 
# rng = np.random.RandomState(0)
# Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
# ynew = model.predict(Xnew)
# 
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
# lim = plt.axis()
# plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
# plt.axis(lim);
# =============================================================================

