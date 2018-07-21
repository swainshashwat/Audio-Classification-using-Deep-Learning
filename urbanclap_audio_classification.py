
# coding: utf-8

# In[61]:


# importing necessary libraries and dependencies
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from sklearn.cross_validation import train_test_split
from sklearn import metrics


# In[2]:


train_data_dir = 'sounds/train/train_sound/'
test_data_dir = 'sounds/test/test_sound/'

# reading the labels
train = pd.read_csv('sounds/labels/train.csv')
test  = pd.read_csv('sounds/labels/test.csv')


# In[3]:


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


# ### Reading train.csv and storing into temp

# In[4]:


# parsing train
temp = train.apply(parser,axis=1,data_dir=train_data_dir)
temp.columns = ['ID','feature']


# In[5]:


# adding Class to 'temp'
temp['Class'] = train['Class']


# In[6]:


type(temp)


# ###  Reading test.csv and storing into temp_test

# In[7]:


# parsing test
temp_test = test.apply(parser, axis=1,data_dir=test_data_dir)
temp_test.columns = ['ID', 'feature']


# In[16]:


temp_test = pd.DataFrame(temp_test)
type(temp_test)


# In[19]:


temp_test.columns = ['mix']


# In[23]:


temp_test.keys()


# In[24]:


temp_test[['ID','feature']] = temp_test['mix'].apply(pd.Series)


# In[28]:


temp_test.drop('mix',axis=1,inplace=True)


# In[32]:


print("\n---------------------train data---------------------")
print(type(temp))
print(temp.head())

print("\n---------------------test data---------------------")
print(type(temp_test))
print(temp_test.head())


print('---------------------Checking for NONE values---------------------')
# checking for NONE values
print(temp[temp.Class.isnull()])

# removing NONE values from temp
temp = temp[temp.Class.notnull()]
temp_test = temp_test[temp_test.notnull()]
#print(temp.ID[temp.label.isnull()])


# In[37]:


temp.Class.unique()


# In[38]:


temp.Class.nunique()


# In[35]:


# Label Encoding the audio data
lb = LabelEncoder()

# converting pd.series into np.array for faster processing
X = np.array(temp.feature.tolist())
y = np.array(temp.Class.tolist())


y = to_categorical(lb.fit_transform(y))


# In[62]:


x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.3)


# ## Building a deep learning model

# In[73]:


num_labels = y.shape[1]
filter_size = 2

def categorical_classifier():
    model = Sequential()

    # input and first hidden layer
    model.add(Dense(input_shape=(40,), units=256, activation='relu', kernel_initializer='uniform'))
    model.add(Dropout(0.5))

    # second hidden layer
    model.add(Dense(units=256,activation='relu',kernel_initializer='uniform'))
    model.add(Dropout(0.5))

    # output layer
    model.add(Dense(units=num_labels, activation='softmax'))

    # compiling our model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    # training the data
    #model.fit(X,y, batch_size=32, epochs=500, validation_split=0.3)
    return model


# In[77]:


# training the data
model.fit(x_train,y_train, batch_size=32, epochs=650, validation_data=(x_test, y_test))

