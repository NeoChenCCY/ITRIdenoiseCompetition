# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 01:07:31 2022

@author: NeoChen
"""

import os, fnmatch
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dense, LSTM, Dropout, \
    Lambda, Input, Multiply, Layer, Conv1D
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, \
    EarlyStopping, ModelCheckpoint
import tensorflow as tf
import soundfile as sf
from wavinfo import WavInfoReader
from random import shuffle, seed
import numpy as np

class myDTLN_model():
    def __init__(self):
        '''
        Constructor
        '''

        # defining default cost function
        self.cost_function = self.snr_cost
        # empty property for the model
        
        #from keras.models import Sequential        
        
        self.model = []
        #self.model = Sequential()
        
        # defining default parameters
        self.fs = 16000
        self.batchsize = 32
        self.len_samples = 15
        self.activation = 'sigmoid'
        self.numUnits = 128
        self.numLayer = 2
        self.blockLen = 512
        self.block_shift = 128
        self.dropout = 0.25
        self.lr = 1e-3
        self.max_epochs = 100 
        self.encoder_size = 256
        self.eps = 1e-7
        # reset all seeds to 42 to reduce invariance between training runs
        os.environ['PYTHONHASHSEED']=str(42)
        seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)
        # some line to correctly find some libraries in TF 2.x
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, enable=True)
                
    @staticmethod
    def snr_cost(s_estimate, s_true):
        '''
        Static Method defining the cost function. 
        The negative signal to noise ratio is calculated here. The loss is 
        always calculated over the last dimension. 
        '''

        # calculating the SNR
        snr = tf.reduce_mean(tf.math.square(s_true), axis=-1, keepdims=True) / \
            (tf.reduce_mean(tf.math.square(s_true-s_estimate), axis=-1, keepdims=True)+1e-7)
        # using some more lines, because TF has no log10
        num = tf.math.log(snr) 
        denom = tf.math.log(tf.constant(10, dtype=num.dtype))
        loss = -10*(num / (denom))
        # returning the loss
        return loss
    
    
    """
    參數群的定義(START)
    
    """
    
    def stftLayer(self, x):
        '''
        Method for an STFT helper layer used with a Lambda layer. The layer
        calculates the STFT on the last dimension and returns the magnitude and
        phase of the STFT.
        '''
        
        # creating frames from the continuous waveform
        frames = tf.signal.frame(x, self.blockLen, self.block_shift)
        # calculating the fft over the time frames. rfft returns NFFT/2+1 bins.
        stft_dat = tf.signal.rfft(frames)
        # calculating magnitude and phase from the complex signal
        mag = tf.abs(stft_dat)
        phase = tf.math.angle(stft_dat)
        # returning magnitude and phase as list
        return [mag, phase]
    
    def seperation_kernel(self, num_layer, mask_size, x, stateful=False):
        '''
        Method to create a separation kernel. 
        !! Important !!: Do not use this layer with a Lambda layer. If used with
        a Lambda layer the gradients are updated correctly.

        Inputs:
            num_layer       Number of LSTM layers
            mask_size       Output size of the mask and size of the Dense layer
        '''

        # creating num_layer number of LSTM layers
        for idx in range(num_layer):
            x = LSTM(self.numUnits, return_sequences=True, stateful=stateful)(x)
            # using dropout between the LSTM layer for regularization 
            if idx<(num_layer-1):
                x = Dropout(self.dropout)(x)
        # creating the mask with a Dense and an Activation layer
        mask = Dense(mask_size)(x)
        mask = Activation(self.activation)(mask)
        # returning the mask
        return mask
    
    def ifftLayer(self, x):
        '''
        Method for an inverse FFT layer used with an Lambda layer. This layer
        calculates time domain frames from magnitude and phase information. 
        As input x a list with [mag,phase] is required.
        '''
        
        # calculating the complex representation
        s1_stft = (tf.cast(x[0], tf.complex64) * 
                    tf.exp( (1j * tf.cast(x[1], tf.complex64))))
        # returning the time domain frames
        return tf.signal.irfft(s1_stft)  
    
    def overlapAddLayer(self, x):
        '''
        Method for an overlap and add helper layer used with a Lambda layer.
        This layer reconstructs the waveform from a framed signal.
        '''

        # calculating and returning the reconstructed waveform
        return tf.signal.overlap_and_add(x, self.block_shift)
    
    def lossWrapper(self):
        '''
        A wrapper function which returns the loss function. This is done to
        to enable additional arguments to the loss function if necessary.
        '''
        def lossFunction(y_true,y_pred):
            # calculating loss and squeezing single dimensions away
            loss = tf.squeeze(self.cost_function(y_pred,y_true))
            # calculate mean over batches
            loss = tf.reduce_mean(loss)
            # return the loss
            return loss
        # returning the loss function as handle
        return lossFunction
    
    """
    參數群的定義(END)
    
    """
    
    def mybuild_DTLN_model(self, norm_stft=False):
        '''
        Method to build and compile the DTLN model. The model takes time domain 
        batches of size (batchsize, len_in_samples) and returns enhanced clips 
        in the same dimensions. As optimizer for the Training process the Adam
        optimizer with a gradient norm clipping of 3 is used. 
        The model contains two separation cores. The first has an STFT signal 
        transformation and the second a learned transformation based on 1D-Conv 
        layer. 
        '''
        
        # input layer for time signal
        time_dat = Input(batch_shape=(None, None))
        # calculate STFT
        mag,angle = Lambda(self.stftLayer)(time_dat)
        # normalizing log magnitude stfts to get more robust against level variations
        if norm_stft:
            mag_norm = InstantLayerNormalization()(tf.math.log(mag + 1e-7))
        else:
            # behaviour like in the paper
            mag_norm = mag
        # predicting mask with separation kernel  
        mask_1 = self.seperation_kernel(self.numLayer, (self.blockLen//2+1), mag_norm)
        # multiply mask with magnitude
        estimated_mag = Multiply()([mag, mask_1])
        # transform frames back to time domain
        estimated_frames_1 = Lambda(self.ifftLayer)([estimated_mag,angle])
        # encode time domain frames to feature domain
        encoded_frames = Conv1D(self.encoder_size,1,strides=1,use_bias=False)(estimated_frames_1)
        # normalize the input to the separation kernel
        encoded_frames_norm = InstantLayerNormalization()(encoded_frames)
        # predict mask based on the normalized feature frames
        mask_2 = self.seperation_kernel(self.numLayer, self.encoder_size, encoded_frames_norm)
        # multiply encoded frames with the mask
        estimated = Multiply()([encoded_frames, mask_2]) 
        # decode the frames back to time domain
        decoded_frames = Conv1D(self.blockLen, 1, padding='causal',use_bias=False)(estimated)
        # create waveform with overlap and add procedure
        estimated_sig = Lambda(self.overlapAddLayer)(decoded_frames)

        
        # create the model
        self.model = Model(inputs=time_dat, outputs=estimated_sig)
        # show the model summary
        print(self.model.summary())
        
    def mytrain_model(self, runName, path_to_train_mix, path_to_train_speech, \
                    path_to_val_mix, path_to_val_speech):
        '''
        Method to train the DTLN model. 
        '''
        
        # create save path if not existent
        savePath = './models_'+ runName+'/' 
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        # create log file writer
        csv_logger = CSVLogger(savePath+ 'training_' +runName+ '.log')
        # create callback for the adaptive learning rate
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=3, min_lr=10**(-10), cooldown=1)
        # create callback for early stopping
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, 
            patience=10, verbose=0, mode='auto', baseline=None)
        # create model check pointer to save the best model
        checkpointer = ModelCheckpoint(savePath+runName+'.h5',
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True,
                                       mode='auto',
                                       save_freq='epoch'
                                       )

        # calculate length of audio chunks in samples
        len_in_samples = int(np.fix(self.fs * self.len_samples / 
                                    self.block_shift)*self.block_shift)
        # create data generator for training data
        generator_input = audio_generator(path_to_train_mix, 
                                          path_to_train_speech, 
                                          len_in_samples, 
                                          self.fs, train_flag=True)
        dataset = generator_input.tf_data_set
        dataset = dataset.batch(self.batchsize, drop_remainder=True).repeat()
        # calculate number of training steps in one epoch
        steps_train = generator_input.total_samples//self.batchsize
        # create data generator for validation data
        generator_val = audio_generator(path_to_val_mix,
                                        path_to_val_speech, 
                                        len_in_samples, self.fs)
        dataset_val = generator_val.tf_data_set
        dataset_val = dataset_val.batch(self.batchsize, drop_remainder=True).repeat()
        # calculate number of validation steps
        steps_val = generator_val.total_samples//self.batchsize
        # start the training of the model
        
        #self.model.add(tf.keras.layers.Conv2D(input_shape = (32,32,3), filters = 8, kernel_size = (5,5),activation = "relu", padding = "same" ))
        
# =============================================================================
#         from keras.models import Sequential
#         self.model = Sequential()
#         self.model.add(Embedding(output_dim=32,
#                             input_dim=2000, 
#                             input_length=100))
#         self.model.add(Dropout(0.2))
#         self.model.add(Flatten())
#         self.model.add(Dense(units=256,
#                         activation='relu' ))
#         self.model.add(Dropout(0.2))
#         self.model.add(Dense(units=1,
#                         activation='sigmoid' ))
#         self.model.summary()        
#         self.model.compile(loss='binary_crossentropy',metrics=['accuracy'])
# =============================================================================

        
        # Method to compile the model for training #        
        # use the Adam optimizer with a clipnorm of 3
        optimizerAdam = keras.optimizers.Adam(lr=self.lr, clipnorm=3.0)
        # compile model with loss function
        self.model.compile(optimizer=optimizerAdam, loss=self.lossWrapper())
        #return self.model
        #self.model.compile(optimizer=optimizerAdam, loss=tf.keras.losses.BinaryCrossentropy(),metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.FalseNegatives()], run_eagerly=False)
        
        # => 最終BUG在此，RUN第二次就掛掉。因為不明參數太多 #
        self.model.fit(
            x=dataset, 
            batch_size=3,
            #steps_per_epoch=steps_train, 
            epochs=self.max_epochs,
            #verbose=1,
            #validation_data=dataset_val,
            validation_steps=steps_val, 
            #callbacks=[checkpointer, reduce_lr, csv_logger, early_stopping],
            #max_queue_size=50,
            #workers=4,
            #use_multiprocessing=True)
            )
        # clear out garbage
        tf.keras.backend.clear_session()
        
    
class InstantLayerNormalization(Layer):
    '''
    Class implementing instant layer normalization. It can also be called 
    channel-wise layer normalization and was proposed by 
    Luo & Mesgarani (https://arxiv.org/abs/1809.07454v2) 
    '''

    def __init__(self, **kwargs):
        '''
            Constructor
        '''
        super(InstantLayerNormalization, self).__init__(**kwargs)
        self.epsilon = 1e-7 
        self.gamma = None
        self.beta = None
        

class audio_generator():
    '''
    Class to create a Tensorflow dataset based on an iterator from a large scale 
    audio dataset. This audio generator only supports single channel audio files.
    '''
    
    def __init__(self, path_to_input, path_to_s1, len_of_samples, fs, train_flag=False):
        '''
        Constructor of the audio generator class.
        Inputs:
            path_to_input       path to the mixtures
            path_to_s1          path to the target source data
            len_of_samples      length of audio snippets in samples
            fs                  sampling rate
            train_flag          flag for activate shuffling of files
        '''
        # set inputs to properties
        self.path_to_input = path_to_input
        self.path_to_s1 = path_to_s1
        self.len_of_samples = len_of_samples
        self.fs = fs
        self.train_flag=train_flag
        # count the number of samples in your data set (depending on your disk,
        #                                               this can take some time)
        self.count_samples()
        # create iterable tf.data.Dataset object 
        self.create_tf_data_obj()
    
    """
    函數方法群(START)
    """
    
    def count_samples(self):
        '''
        Method to list the data of the dataset and count the number of samples. 
        '''

        # list .wav files in directory
        self.file_names = fnmatch.filter(os.listdir(self.path_to_input), '*.wav')
        # count the number of samples contained in the dataset
        self.total_samples = 0
        for file in self.file_names:
            info = WavInfoReader(os.path.join(self.path_to_input, file))
            self.total_samples = self.total_samples + \
                int(np.fix(info.data.frame_count/self.len_of_samples))
    
    def create_tf_data_obj(self):
        '''
        Method to to create the tf.data.Dataset. 
        '''

        # creating the tf.data.Dataset from the iterator
        self.tf_data_set = tf.data.Dataset.from_generator(
                        self.create_generator,
                        (tf.float32, tf.float32),
                        output_shapes=(tf.TensorShape([self.len_of_samples]), \
                                       tf.TensorShape([self.len_of_samples])),
                        args=None
                        )
        
    def create_generator(self):
        '''
        Method to create the iterator. 
        '''

        # check if training or validation
        if self.train_flag:
            shuffle(self.file_names)
        # iterate over the files  
        for file in self.file_names:
            # read the audio files
            noisy, fs_1 = sf.read(os.path.join(self.path_to_input, file))
            speech, fs_2 = sf.read(os.path.join(self.path_to_s1, file))
            # check if the sampling rates are matching the specifications
            if fs_1 != self.fs or fs_2 != self.fs:
                raise ValueError('Sampling rates do not match.')
            if noisy.ndim != 1 or speech.ndim != 1:
                raise ValueError('Too many audio channels. The DTLN audio_generator \
                                 only supports single channel audio data.')
            # count the number of samples in one file
            num_samples = int(np.fix(noisy.shape[0]/self.len_of_samples))
            # iterate over the number of samples
            for idx in range(num_samples):
                # cut the audio files in chunks
                in_dat = noisy[int(idx*self.len_of_samples):int((idx+1)*
                                                        self.len_of_samples)]
                tar_dat = speech[int(idx*self.len_of_samples):int((idx+1)*
                                                        self.len_of_samples)]
                # yield the chunks as float32 data
                yield in_dat.astype('float32'), tar_dat.astype('float32')
    
    """
    函數方法群(END)
    """