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


# =============================================================================
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0, 1))# 
# normalized_stft = scaler.transform(stft)
# scaler.fit(stft)
# features_convolution = np.reshape(normalized_stft,(400,1025, -1,1))
#
# model = Sequential()
# 
# model.add(Conv2D(16, (3, 3), input_shape=features_convolution.shape[1:]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# #'''
# #model.add(Dropout(0.2))
# 
# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# 
# #model.add(Dropout(0.2))
# 
# #'''
# #'''
# model.add(Conv2D(64, (3, 3),padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# #'''
# 
# 
# model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
# 
# #model.add(Dense(1000))#input_shape=features.shape[1:]
# model.add(Dense(64))#input_shape=features.shape[1:]
# 
# model.add(Dense(10))
# model.add(Activation('softmax'))
# sgd = optimizers.SGD(lr=0.0000001, decay=1e-6, momentum=0.9, nesterov=True)
# 
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
# =============================================================================
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(Zxx2.astype(float))
normalized_stft = scaler.transform(Zxx2.astype(float))
features_convolution = np.reshape(normalized_stft,(501,101, -1,1))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from keras import optimizers
import keras

model = Sequential()

features_convolution_float = features_convolution/1000
#model.add(Conv2D(16, (3, 3), input_shape=(features_convolution.shape[1:])))
#model.add(Conv2D(16, (3, 3), input_shape=(features_convolution.reshape(-1,1).astype(float))))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#'''
model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(1000))#input_shape=features.shape[1:]
model.add(Dense(64))#input_shape=features.shape[1:]

model.add(Dense(10))
model.add(Activation('softmax'))
#sgd = optimizers.SGD(lr=0.0000001, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#y=keras.utils.to_categorical(labels, num_classes=10, dtype='float32')
history = model.fit(features_convolution, None,batch_size=8, epochs=40,validation_split=0.2)