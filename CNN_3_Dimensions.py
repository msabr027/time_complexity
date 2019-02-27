# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 10:01:50 2019

@author: mosab
"""

#to call at the beginning
#import sys
#sys.path.insert(0, r'YOURPATH_LOCATION_FOR_CE.py')

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from CE_3_Dimensions import ComplexityEvaluator 


def random_data_classification_image(n, p, l):
    return np.random.randint(low=0,high=255,size=(n,p,p)), np.random.randint(low=0,high=5,size=n), np.random.choice(unit_layer_fix,size=5)

def build_model(hidden_layer_sizes,inp,nout):
    model = Sequential()
    model.add(Conv2D(hidden_layer_sizes[0], (3,3), activation='relu', input_shape=(inp.shape[1],inp.shape[2],1),data_format='channels_last'))
    for layer_size in hidden_layer_sizes[1:]:
        model.add(Conv2D(layer_size, (1,1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(nout, activation='softmax'))
    print(model.summary)
    return model

#you could setup an evolving number of units per layer architeture to better understand the impact on time complexity of adding units.
#unit_layer = [32, 64, 128, 256, 512, 1024]

unit_layer_fix = 64

complexity_evaluator = ComplexityEvaluator(
    [500, 25000,400000,800000,4000000],
    [5,100,200,400,1000],[1,5,15,30,50])      

res = complexity_evaluator.Run(random_data_classification_image)[0]
print(res)