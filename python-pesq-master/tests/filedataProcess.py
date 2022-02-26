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
from scipy import signal

from pathlib import Path

from pesq import pesq, NoUtterancesError, PesqError

import matplotlib.pyplot as plt
import numpy as np

# read file #
data_dir = Path(__file__).parent.parent / 'audio'
ref_path = data_dir / 'speech.wav'
deg_path = data_dir / 'speech_bab_0dB.wav'
sample_rate1, ref = scipy.io.wavfile.read(ref_path)
sample_rate2, deg = scipy.io.wavfile.read(deg_path)

f1, t1, Zxx1 = signal.stft(ref, sample_rate1, nperseg=1000)
f2, t2, Zxx2 = signal.stft(deg, sample_rate2, nperseg=1000)

# =============================================================================
# #amp = 2 * np.sqrt(1000)
# plt.pcolormesh(t1, f1, np.abs(Zxx1), vmin=0, vmax=None, shading='gouraud')
# plt.title('STFT Magnitude of speech.wav')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()
# =============================================================================
plt.pcolormesh(t2, f2, np.abs(Zxx2), vmin=0, vmax=None, shading='gouraud')
plt.title('STFT Magnitude of speech_bab_0dB.wav')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

