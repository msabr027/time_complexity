# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 12:29:36 2019

@author: Mohamed Sabri

@ref : https://www.thekerneltrip.com/machine/learning/computational-complexity-learning-algorithms/
"""

import time
import gc
import pandas as pd
import numpy as np
from keras.backend import clear_session
from sklearn.linear_model import LinearRegression


class ComplexityEvaluator:

    def __init__(self, number_rows, number_columns, number_layers):
        self._nrow_samples = number_rows
        self._ncol_samples = number_columns
        self._nlay_samples = number_layers

    def _time_samples(self, random_data_generator):
        rows_list = []
        for nrow in self._nrow_samples:
            for ncol in self._ncol_samples:
                for nlay in self._nlay_samples:
                    gc.collect()
                    train, labels, layers = random_data_generator(nrow, ncol, nlay)
                    class_uniq = pd.DataFrame(labels).nunique()
                    labels  = pd.get_dummies(labels,prefix=['Class'])
                    train = train.reshape(train.shape[0],train.shape[1],train.shape[2],1)
                    model = build_model(layers,train,class_uniq[0])
                    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                    start_time = time.time()
                    model.fit(train, labels,epochs=1, batch_size=5)
                    elapsed_time = time.time() - start_time
                    result = {"Rows": nrow,"Features": ncol,"Layers": nlay,"Time": elapsed_time}
                    rows_list.append(result)
                    clear_session() 
                    
        return rows_list
    
    def Run(self, random_data_generator):
        data = pd.DataFrame(self._time_samples(random_data_generator))
        print(data)
        data = data.apply(np.log)
        linear_model = LinearRegression(fit_intercept=True)
        linear_model.fit(data[["Rows", "Features","Layers"]], data[["Time"]])
        return linear_model.coef_
