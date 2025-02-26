# SIM-WS20
This repository holds applications that use different similarity modelling approaches to identify different characters of the muppet show by audio and video.

# Team Members
Martin Ledl, 01634019
Alexander Selzer, 01633655


# Introduction & Instructions
**Note: This Readme aims for given a short overview of the structure. More detailed information and documentation is provided in the corresponding notebooks - SIM/src/SIM1.ipynb and SIM/src/SIM2.ipynb **
The solutions for the respective courses SIM1 and SIM2 can be found under SIM/src/SIM1.ipynb and SIM/src/SIM2.ipynb. For SIM1 we
decided to go with feature engineering in combination with shallow classifier and we are going to tackle SIM2 with Neural Networks
(CNNs and RNNs). For this first attempts we tried to go with the recommendations from the lectures. Therefore, we were going for
colour features for Kermit the frog and for audio using MFCC features for Waldorf & Statler.
The major work for SIM in general was extracting images for a ground truth which can be used for training and testing.
Currently, tain, test and validation split is done on the extracted data, but we are planning to have a dedicated validation set
which we will do our final testing against and not just splitting it of the dataset.
The datasets (images and audio) which are used for training consist of 50% positive instances (all positive occurrences across) 
and 50% negative samples which are randomly sampled across all 3 videos with respect to the data distribution.

Before executing the respective jupyter notebooks, just put the 3 corresponding episodes of the Muppet Show in the "videos" folder.
The datasets for training and testing are the automatically extracted if the notebook is run.
The textfiles which hold the labels are stored under "ground_truth/'videoname'/'videoname'.txt", for example "ground_truth/Muppets-02-01-01/Muppets-02-01-01.txt".

# File Descriptions
labeler.py: Small Python program we used to semi-automatically label the the video. This generates labeled images as well as a textfile with 
the labels per frame for the labeled video.
audio_extractor.py: Provides utils for extacting audio from videos.
image_extractor.py: Provides utils for extacting images from videos.
dataset_generator.py: Provides functionality for extracting the datasets for the specific prediction tasks. Those functionalities
are used by the jupyter notebooks.

# SIM 2 models

Due to the long amount of time required to train the CNN models, we have uploaded the trained models to owncloud[https://owncloud.tuwien.ac.at/index.php/s/ZgPcnRf8MOk8dFk](https://owncloud.tuwien.ac.at/index.php/s/ZgPcnRf8MOk8dFk)

## pig-model-1

A CNN based on VGG16 for pig classification (~96% accuracy on the eval set).

## rnn-model

An RNN based on GRU layers for detecting the swedish chef in audio. It achieves 85% accuracy on the test set and 76% accuracy on the eval set. The approach was limited by the amount of ddata we had for the swedish chef, however the model could be trained quite effectively and we believe it could achieve good results if more data was available.
