# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 12:29:36 2019

@author: Mohamed Sabri
"""

import time
import gc
import pandas as pd
from keras.backend import clear_session
from sklearn.linear_model import LinearRegression


class ComplexityEvaluator:

    def __init__(self, number_rows, number_columns, number_classes, number_layers, number_drop):
        self._nrow_samples = number_rows
        self._ncol_samples = number_columns
        self._ncls_samples = number_classes
        self._nlay_samples = number_layers
        self._drop_samples = number_drop

    def _time_samples(self, random_data_generator):
        rows_list = []
        for nrow in self._nrow_samples:
            for ncol in self._ncol_samples:
                for ncls in self._ncls_samples:
                    for nlay in self._nlay_samples:
                        for drop in self._drop_samples:
                            gc.collect()
                            train, labels, layers = random_data_generator(nrow, ncol, ncls, nlay)
                            class_uniq = pd.DataFrame(labels).nunique()
                            labels = pd.get_dummies(labels,prefix=['Class'])
                            model = build_model(layers,train,class_uniq[0],drop)
                            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                            start_time = time.time()
                            model.fit(train, labels,epochs=1, batch_size=5)
                            elapsed_time = time.time() - start_time
                            result = {"Rows": nrow,"Features": ncol,"Class": ncls,"Layers": nlay,"Drop": drop,"Time": elapsed_time}
                            rows_list.append(result)
                            clear_session() 

        return rows_list
    
    def Run(self, random_data_generator):
        data = pd.DataFrame(self._time_samples(random_data_generator))
        print(data)
        data = data.apply(np.sqrt)
        linear_model = LinearRegression(fit_intercept=True)
        linear_model.fit(data[["Rows", "Features","Class","Layers","Drop"]], data[["Time"]])
        return linear_model.coef_