'''
Feature extraction and clustering using pre-trained CNN models
'''

# for loading/processing the images  
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 

# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle
import time

path = r"C:\Users\heisu\OneDrive\ドキュメント\STREETS\data\Delft_NL\imagedb"
# change the working directory to the path where the images are located
os.chdir(path)

# this list holds all the image filename
images = []

# creates a ScandirIterator aliased as files
with os.scandir(path) as files:
  # loops through each file in the directory
    for file in files:
        if file.name.endswith('.png'):
          # adds only the image files to the flowers list
            images.append(file.name)

# exclude images with _s_a and _s_b
images = [x for x in images if not x.endswith('_s_a.png')]
images = [x for x in images if not x.endswith('_s_b.png')]

# test load the first image
# original size of the image == (600, 900, 3)
# VGG model takes input of size (224, 224, 3)
# img = load_img(images[0], target_size=(224, 224))
# img = np.array(img)
# # print(img.shape)

# reshaped_img = img.reshape(1,224,224,3)
# # print(reshaped_img.shape)

# # prepare the image for the VGG model
# x = preprocess_input(reshaped_img)   # numpy array
# print(x.shape)

# load the VGG16 model
model = VGG16()
# remove the output layer
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

def extract_features(file, model):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224,224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img) 
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3) 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    
    return features

# extract the features from all images
data = {}
p = r"C:\Users\heisu\OneDrive\ドキュメント\STREETS\data\Delft_NL\features/features.pkl"

img_sample = images[0:100]

start_time = time.time()
# lop through each image in the dataset
for image in img_sample:
    # try to extract the features and update the dictionary
    try:
        feat = extract_features(image,model)
        data[image] = feat
    # if something fails, save the extracted features as a pickle file (optional)
    except:
        with open(p,'wb') as file:
            pickle.dump(data,file)
end_time = time.time()
print("Time taken in seconds: ", end_time - start_time)
# get a list of the filenames
filenames = np.array(list(data.keys()))
# get a list of just the features
feat = np.array(list(data.values()))
print(filenames.shape, feat.shape)

