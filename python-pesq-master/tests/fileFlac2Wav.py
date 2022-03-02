# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 20:42:12 2022

@author: NeoChen
"""
from pathlib import Path
import os
import soundfile as sf

data_dir = Path(__file__).parent.parent / 'DATA/test'
data_dir_files = os.listdir(data_dir)
data_dir_save = Path(__file__).parent.parent / 'DATA/test2wav'

for i_name in data_dir_files:
    # 轉檔 flac to wav #
    degIN_path = data_dir / i_name 
    degSAVE_path = data_dir_save / i_name.replace(".flac", ".wav")
    deg, sample_rate = sf.read(degIN_path)
    sf.write(degSAVE_path, deg, 16000, 'PCM_24')