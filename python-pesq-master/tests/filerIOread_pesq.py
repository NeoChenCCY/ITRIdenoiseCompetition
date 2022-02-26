# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 15:28:21 2022

@author: Administrator

[note]
1. score check ok
2. IOread and plot check ok
3. denoise ?
4. model tran ?

"""
import pytest
import numpy as np

import scipy.io.wavfile
import soundfile as sf     

from pathlib import Path

from pesq import pesq, NoUtterancesError, PesqError

import matplotlib.pyplot as plt
import numpy as np

data_dir = Path(__file__).parent.parent / 'audio'
ref_path = data_dir / 'speech.wav'
deg_path = data_dir / 'speech_bab_0dB.wav'
#deg_path = data_dir / 'mixed_00006_dog_bark.wav'
#degSAVE_path = data_dir / 'mixed_00006_dog_bark.wav'

sample_rate1, ref = scipy.io.wavfile.read(ref_path)
sample_rate2, deg = scipy.io.wavfile.read(deg_path)

#deg, sample_rate = sf.read(deg_path)
#sf.write(degSAVE_path, deg, 16000, 'PCM_24')

score1 = pesq(ref=ref, deg=deg, fs=sample_rate1, mode='wb')

#assert score == 1.0832337141036987, score

score2 = pesq(ref=ref, deg=deg, fs=sample_rate2, mode='nb')

#assert score == 1.6072081327438354, score

duration = len(ref)/sample_rate1
time = np.arange(0,duration,1/sample_rate1) #time vector

plt.plot(time,ref)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('speech.wav')
plt.show()

duration = len(deg)/sample_rate2
time = np.arange(0,duration,1/sample_rate2) #time vector

plt.plot(time,deg)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('speech_bab_0dB.wav')
plt.show()

tmp = deg - ref

duration = len(tmp)/sample_rate2
time = np.arange(0,duration,1/sample_rate2) #time vector

plt.plot(time,tmp)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('tmp.wav')
plt.show()

scipy.io.wavfile.write("tmp.wav", sample_rate2, tmp)

score3 = pesq(ref=deg, deg=ref, fs=sample_rate2, mode='wb')