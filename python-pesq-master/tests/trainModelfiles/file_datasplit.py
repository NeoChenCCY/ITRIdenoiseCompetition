# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 06:10:02 2022

@author: NeoChen
"""
# # Creating Train / Val / Test folders (One time use)
import os
import numpy as np
import shutil
import random
from pathlib import Path

#root_dir = '../classification/data/' # data root path
#root_dir = Path(__file__).parent.parent / 'train/'
root_dir = '../train/' # data root path
classes_dir = ['mixed2wav', 'vocal2wav'] #total labels

#root_dir_str = str(root_dir)

val_ratio = 0.15
test_ratio = 0.05

for cls in classes_dir:
    os.makedirs(root_dir +'train/' + cls)
    os.makedirs(root_dir +'val/' + cls)
    os.makedirs(root_dir +'test/' + cls)


# Creating partitions of the data after shuffeling
src = root_dir + cls # Folder to copy images from

allFileNames = os.listdir(src)
np.random.shuffle(allFileNames)
train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                          [int(len(allFileNames)* (1 - (val_ratio + test_ratio))), 
                                                           int(len(allFileNames)* (1 - test_ratio))])


train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

print('Total images: ', len(allFileNames))
print('Training: ', len(train_FileNames))
print('Validation: ', len(val_FileNames))
print('Testing: ', len(test_FileNames))

# Copy-pasting images
for name in train_FileNames:
    shutil.copy(name, root_dir +'train/' + cls)
    print(root_dir + 'train/' + name)
    
for name in val_FileNames:
    shutil.copy(name, root_dir +'val/' + cls)
    print(root_dir + 'train/' + name)

for name in test_FileNames:
    shutil.copy(name, root_dir +'test/' + cls)
    print(root_dir + 'train/' + name)
