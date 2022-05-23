import random

import numpy as np
import scipy.io as sio


def normalization_unit(data):
    # unit every rows in the data ( n_sample * n_feature )
    data = np.mat(data)
    normalized_data = np.divide(data, np.sqrt(np.sum(np.multiply(data, data), axis=1)))
    return normalized_data

def AR(seed, train_num):
    random.seed(seed)
    random_index = random.sample(range(1400), 1400)
    train_index = random_index[0:train_num]
    test_index = random_index[train_num:1400]
    
    data=sio.loadmat("data/AR_55_40.mat")
    labels = data['labels'].T
    labels = labels.flatten()
    images = data['DAT'].T
    images = normalization_unit(images)
    print("class num:", len(np.unique(labels)))
    print("image size", images.shape)
    
    train_data = images[train_index]
    test_data = images[test_index]
    train_labels = labels[train_index]
    test_labels = labels[test_index]
    
    return train_data, train_labels, test_data, test_labels


def Yale(seed, train_num): 
    random.seed(seed)
    random_index = random.sample(range(2204), 2204)
    train_index = random_index[0:train_num]
    test_index = random_index[train_num:2204]
    
    data=sio.loadmat("data/ExYaleB_54_48.mat")
    labels = data['labels'].T
    labels = labels.flatten()
    images = data['DAT'].T
    images = normalization_unit(images)
    print("class num:", len(np.unique(labels)))
    print("image size", images.shape)
   
    train_data = images[train_index]
    test_data = images[test_index]
    train_labels = labels[train_index]
    test_labels = labels[test_index]
    
    return train_data, train_labels, test_data, test_labels