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
degtest_path = data_dir / 'mixed_01081_jackhammer.wav'
# =============================================================================
#ref_path = data_dir / 'vocal_01081.wav'
#deg_path = data_dir / 'mixed_01081_jackhammer.wav'
sample_rate1, ref = scipy.io.wavfile.read(ref_path)
sample_rate2, deg = scipy.io.wavfile.read(deg_path)

sample_rate3, degtest = scipy.io.wavfile.read(degtest_path)

f1, t1, Zxx1 = signal.stft(ref, sample_rate1, nperseg=1000)
f2, t2, Zxx2 = signal.stft(deg, sample_rate2, nperseg=1000)

f3, t3, Zxx3 = signal.stft(degtest, sample_rate2, nperseg=1000)

# plot voice data #
# =============================================================================
# plt.scatter(x,y1,c="red")
# plt.scatter(x,y2,c="green")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.title("Scatter Plot of two different datasets")
# plt.show()
# =============================================================================
Zxx5 = Zxx2 - Zxx1
diff_dr = deg - ref
#X4, y4 = make_blobs(500, 133, centers=2, random_state=2, cluster_std=1.5)

# 錯誤範例 #
# =============================================================================
# plt.scatter(deg,ref,c="red")
# plt.scatter(deg,diff_dr,c="green")
# plt.scatter(deg,deg,c="yellow")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.title("Scatter Plot of two different datasets")
# plt.show()
# 
# plt.scatter(Zxx2,Zxx1,c="red")
# plt.scatter(Zxx2,Zxx5,c="green")
# plt.scatter(Zxx2,Zxx2,c="yellow")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.title("Scatter Plot of two different datasets")
# plt.show()
# =============================================================================

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
# 轉換型別
# Xs_new = np.dot(Xs, A_coral).astype(float)
#
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
# lim = plt.axis()
# plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
# plt.axis(lim);
# =============================================================================
"""
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

Zxx_train = Zxx2.reshape(-1, 1)[0:int(len(Zxx2.reshape(-1, 1))*0.80)].astype(float)
model.fit(Zxx2.reshape(-1, 1).astype(float), Zxx1.reshape(-1, 1).astype(int));
Zxx_test = Zxx2.reshape(-1, 1)[int(len(Zxx2.reshape(-1, 1))*0.80):len(Zxx2.reshape(-1, 1))].astype(float)
ynew = model.predict(Zxx_test)

plt.scatter(ynew.reshape(-1, 1),None,c="red")
#plt.scatter(ynew,diff_dr,c="green")
plt.scatter(ynew.reshape(-1, 1),ynew,c="yellow")
#plt.scatter(ynew.reshape(-1, 1),Zxx2.reshape(-1, 1)[int(len(Zxx2.reshape(-1, 1))*0.80):len(Zxx2.reshape(-1, 1))].astype(float),c="yellow")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter Plot of two different datasets")
plt.show()
"""

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(Zxx1.reshape(-1, 1).astype(int), Zxx1.reshape(-1, 1).astype(int));

Zxx_test = Zxx2.reshape(-1, 1).astype(float)
ynew = model.predict(Zxx_test)


plt.scatter(ynew.reshape(-1, 1), Zxx1.reshape(-1, 1), c="red")
#plt.scatter(deg,ref.reshape(-1, 1),c="red")
#plt.scatter(ynew.reshape(-1, 1), ynew, c="yellow")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter Plot of two different datasets")
plt.show()

#plt.scatter(ynew.reshape(-1, 1), ref.reshape(-1, 1), c="green")
plt.scatter(Zxx2,Zxx1.reshape(-1, 1),c="green")
#plt.scatter(ynew.reshape(-1, 1), ynew, c="yellow")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter Plot of two different datasets")
plt.show()

tmp = Zxx2.reshape(-1, 1) - Zxx1.reshape(-1, 1)
plt.scatter(tmp.reshape(-1, 1), Zxx1.reshape(-1, 1), c="yellow")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter Plot of two different datasets")
plt.show()

"""
tmp = deg.reshape(-1, 1) - ref.reshape(-1, 1)
plt.scatter(tmp.reshape(-1, 1), ref.reshape(-1, 1), c="yellow")
plt.scatter(deg,ref.reshape(-1, 1),c="green")
plt.scatter(ynew.reshape(-1, 1), ref.reshape(-1, 1), c="red")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter Plot of two different datasets")
plt.show()
"""

tmp = Zxx2.reshape(-1, 1) - Zxx1.reshape(-1, 1)
#plt.scatter(ref.reshape(-1, 1), tmp.reshape(-1, 1), c="yellow")
plt.scatter(Zxx2,Zxx1.reshape(-1, 1),c="green")
plt.scatter(ynew.reshape(-1, 1), Zxx1.reshape(-1, 1), c="red")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter Plot of two different datasets")
plt.show()

"""
model = GaussianNB()

model.fit(Zxx2.reshape(-1, 1).astype(float), Zxx2.reshape(-1, 1).astype(int));

Zxx_test = Zxx1.reshape(-1, 1).astype(float)
ynew = model.predict(Zxx_test)


plt.scatter(ynew.reshape(-1, 1), ref.reshape(-1, 1), c="red")
#plt.scatter(deg,ref.reshape(-1, 1),c="red")
#plt.scatter(ynew.reshape(-1, 1), ynew, c="yellow")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter Plot of two different datasets")
plt.show()

#plt.scatter(ynew.reshape(-1, 1), ref.reshape(-1, 1), c="green")
plt.scatter(deg,ref.reshape(-1, 1),c="green")
#plt.scatter(ynew.reshape(-1, 1), ynew, c="yellow")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter Plot of two different datasets")
plt.show()


plt.scatter(deg,ref.reshape(-1, 1),c="green")
plt.scatter(ynew.reshape(-1, 1), ref.reshape(-1, 1), c="red")
#plt.scatter(ynew.reshape(-1, 1), ynew, c="yellow")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter Plot of two different datasets")
plt.show()
"""