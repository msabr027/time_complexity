# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 16:16:17 2019

@author: Mohamed Sabri

@ref : https://www.thekerneltrip.com/machine/learning/computational-complexity-learning-algorithms/
"""

#to call at the beginning
#import sys
#sys.path.insert(0, r'YOURPATH_LOCATION_FOR_CE.py')


import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from CE_5_Dimensions import ComplexityEvaluator 



        
def random_data_classification(n, p, o, l):
    return np.random.rand(n, p), np.random.randint(low=0,high=o,size=n), np.random.randint(low=2,high=10,size=l)

def build_model(hidden_layer_sizes,inp,nout,dropout):
    model = Sequential()
    model.add(Dense(hidden_layer_sizes[0], input_dim=inp.shape[1], kernel_initializer= 'uniform', activation = 'relu'))
    model.add(Activation('tanh'))
    if dropout==0 :
        for layer_size in hidden_layer_sizes[1:]:
            model.add(Dense(layer_size))
            model.add(Activation('tanh'))
        model.add(Dense(nout))
        model.add(Activation('sigmoid'))
        print(model.summary)
        return model
    else :
        for layer_size in hidden_layer_sizes[1:]:
            model.add(Dense(layer_size))
            model.add(Activation('tanh'))
            model.add(Dropout(0.2))
        model.add(Dense(nout))
        model.add(Activation('sigmoid'))  
        print(model.summary)         
        return model
    return model


#complexity_evaluator = ComplexityEvaluator(
    #[500, 1000, 2000, 5000, 10000, 15000, 20000],
    #[5, 10, 20, 50, 100, 200],[2,5,10,20,50],[1,3,5,10,20],[0,1])    

complexity_evaluator = ComplexityEvaluator(
    [500, 5000, 15000, 25000,200000],
    [5, 200],[2,50],[1,20],[0,1])   

res = complexity_evaluator.Run(random_data_classification)[0]
print(res)
