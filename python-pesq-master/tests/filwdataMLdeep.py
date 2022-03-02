# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 07:53:25 2022

@author: Administrator
"""

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


# deep ML CNN#
# =============================================================================
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation,Flatten
# from keras.layers.embeddings import Embedding
# 
# model = Sequential()
# model.add(Embedding(output_dim=32,
#                     input_dim=2000, 
#                     input_length=100))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(units=256,
#                 activation='relu' ))
# model.add(Dropout(0.2))
# model.add(Dense(units=1,
#                 activation='sigmoid' ))
# model.summary()
# 
# model.compile(loss='binary_crossentropy',metrics=['accuracy'])
# #進行訓練
# #batch_size：每一批次訓練100筆資料
# #epochs：執行10個訓練週期
# #verbose：顯示每次的訓練過程
# #validation_split：測試資料的比例
# train_history =model.fit(x_train, y_train,batch_size=100,
#                          epochs=10,verbose=2,validation_split=0.25)
#                       
# #評估訓練模型的準確率
# acu = model.evaluate(x_test, y_test, verbose=1)
# acu[1]
# =============================================================================

# ML LinearRegression #
# =============================================================================
# from sklearn.linear_model import LinearRegression
# model = LinearRegression(fit_intercept=True)
# 
# model.fit(x[:, np.newaxis], y)
# 
# xfit = np.linspace(0, 10, 1000)
# yfit = model.predict(xfit[:, np.newaxis])
# 
# plt.scatter(x, y)
# plt.plot(xfit, yfit);
# =============================================================================


from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)

model.fit(Zxx1.reshape(-1, 1).astype(int), Zxx1.reshape(-1, 1).astype(int))

xfit = Zxx2.reshape(-1, 1).astype(float)
yfit = model.predict(xfit)

plt.scatter(Zxx1.reshape(-1, 1).astype(int), Zxx1.reshape(-1, 1).astype(int))
plt.plot(xfit, yfit);