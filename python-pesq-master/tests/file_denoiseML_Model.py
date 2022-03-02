#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:23:15 2020

@author: nils
"""

import soundfile as sf
import numpy as np
import tensorflow as tf

from pathlib import Path
from pesq import pesq
import scipy.io.wavfile

##########################
# the values are fixed, if you need other values, you have to retrain.
# The sampling rate of 16k is also fix.
block_len = 512
block_shift = 128


data_dir = Path(__file__).parent.parent / 'audio'
#ref_path = data_dir / 'speech.wav'
#deg_path = data_dir / 'speech_bab_0dB.wav'
#degIN_path = data_dir / 'vocal_01081.flac'
degSAVE_path = data_dir / 'vocal_01081.wav'
data_dir = Path(__file__).parent.parent / 'audio'
ref_path = data_dir / 'mixed_01081_jackhammer.wav'

# load model
model = tf.saved_model.load('./pretrained_model/dtln_saved_model')
infer = model.signatures["serving_default"]
# load audio file at 16k fs (please change)
audio,fs = sf.read(ref_path)
# check for sampling rate
if fs != 16000:
    raise ValueError('This model only supports 16k sampling rate.')
# preallocate output audio
out_file = np.zeros((len(audio)))
# create buffer
in_buffer = np.zeros((block_len))
out_buffer = np.zeros((block_len))
# calculate number of blocks
num_blocks = (audio.shape[0] - (block_len-block_shift)) // block_shift
# iterate over the number of blcoks        
for idx in range(num_blocks):
    # shift values and write to buffer
    in_buffer[:-block_shift] = in_buffer[block_shift:]
    in_buffer[-block_shift:] = audio[idx*block_shift:(idx*block_shift)+block_shift]
    # create a batch dimension of one
    in_block = np.expand_dims(in_buffer, axis=0).astype('float32')
    # process one block
    out_block= infer(tf.constant(in_block))['conv1d_1']
    # shift values and write to buffer
    out_buffer[:-block_shift] = out_buffer[block_shift:]
    out_buffer[-block_shift:] = np.zeros((block_shift))
    out_buffer  += np.squeeze(out_block)
    # write block to output file
    out_file[idx*block_shift:(idx*block_shift)+block_shift] = out_buffer[:block_shift]
    
    
refOUT_path = data_dir / 'out.wav'
# write to .wav file 
#sf.write('out.wav', out_file, fs)
sf.write(refOUT_path, out_file, fs)

print('Processing finished.')


sample_rate1, ref = scipy.io.wavfile.read(degSAVE_path)
sample_rate2, deg = scipy.io.wavfile.read(refOUT_path)
score1 = pesq(ref=ref, deg=deg, fs=sample_rate1, mode='wb')
