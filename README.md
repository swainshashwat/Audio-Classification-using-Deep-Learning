# Audio-Classification-using-Deep-Learning
Classifying 10 different categories of Urban Sounds using Deep Learning.

The audio files can be downloaded from the following link: 
https://drive.google.com/drive/folders/0By0bAi7hOBAFUHVXd1JCN3MwTEU


## IMPORTANT: The folders should be arranged in the following manner: 
Dir of train label: sounds/labels/train.csv

Dir of test label: sounds/labels/test.csv

Dir of train sounds:sounds/train/train_sound/ (audio files in .wav format)

Dir of train sounds:sounds/test/test_sound/ (audio files in .wav format)


### The train folder are labelled
### The test folder aren't labelled

We separate one audio signal into 3 to actually load the data into a machine understandable format. 
For this, we simply take values after every specific time steps. 
For example; in a 2 second audio file, we extract values at half a second. 
![Alt Text](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/08/23210623/sound.png)
This is called sampling of audio data, and the rate at which it is sampled is called the sampling rate.

Different pure signals, which can now be represented as three unique values in frequency domain.

There are a few more ways in which audio data can be represented, for example. using MFCs (Mel-Frequency cepstrums).
These are nothing but different ways to represent the data.

Next we extract features from this audio representations, so that our Deep Learning model can work on these features and perform the task it is designed for..
