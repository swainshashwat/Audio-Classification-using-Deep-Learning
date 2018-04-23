 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 22:00:37 2018

@author: astaroth
"""

# importing necessary libraries and dependencies
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from sklearn import metrics

train_data_dir = 'sounds/train/train_sound/'
test_data_dir = 'sounds/test/test_sound/'

# reading the labels
train = pd.read_csv('sounds/labels/train.csv')
test  = pd.read_csv('sounds/labels/test.csv')

# function to load files and extract features
def parser(row, data_dir):
    # setting path
    file_name = os.path.join(data_dir,str(row.ID)+'.wav')
    print(file_name)
    # check if the file is corrupted
    try:
        # here kaiser_fast is a technique used for faster extraction
        # X-> audio_time_series_data; sample_rate-> sampling rate
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        
        # extraccting Mel-Frequeny Cepstral Coeficients feature from data
        # y -> accepts time-series audio data; sr -> accepts sampling rate
        # n_mfccs -> no. of MFCCs to return
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis = 0)
    
    except Exception as e:
        print("Error encountered while parsing file: ", e)
        return None, None
    
    # store mfccs features
    feature = mfccs
    # store the respective id
    data_id = row.ID
    
    return [data_id, feature]

# storing the audio files in DataFrame 'temp' and 'temp_test'

# parsing train
temp = train.apply(parser,axis=1,data_dir=train_data_dir)
temp.columns = ['ID','feature']

# adding Class to 'temp' and 'temp_test'
temp['Class'] = train['Class']

# parsing test
temp_test = test.apply(parser, axis=1,data_dir=test_data_dir)
temp_test.columns = ['ID', 'feature']

print("\ntrain data")
print(temp.head())

print("\ntest data")
print(temp_test.head())

# checking for NONE values
print(temp.ID[temp.label.isnull()])
print(temp_test.ID[temp.label.isnull()])

# removing NONE values from temp
temp = temp[temp.label.notnull()]
temp_test = temp_test[temp_test.notnull()]
#print(temp.ID[temp.label.isnull()])

# Label Encoding the audio data

lb = LabelEncoder()


# converting pd.series into np.array for faster processing
X = np.array(temp.feature.tolist())
y = np.array(temp.label.tolist())


y = to_categorical(lb.fit_transform(y))


# building a deep learning model
num_labels = y.shape[1]
filter_size = 2

model = Sequential()

# input and first hidden layer
model.add(Dense(input_dim=40, units=256, activation='relu'))
model.add(Dropout(0.5))

# second hidden layer
model.add(Dense(units=256,activation='relu'))
model.add(Dropout(0.5))

# output layer
model.add(Dense(output_dim=num_labels, activation='softmax'))

# compiling our model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# training the data
model.fit(X,y, batch_size=32, epochs=5, validation_data=(val_x, val_y))