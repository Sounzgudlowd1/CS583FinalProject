# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 18:37:28 2018

@author: Erik
"""

from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.regularizers import l1, l2, l1_l2
import numpy as np
from get_data import get_data_custom, one_hot_encode


np.random.seed(7)
#file name, max gram length, min occurances of gram, remove stop words(T/F)
#X, y = get_data_custom('data-1_train.csv', 2, 0, False) results in roughly 72% accuracy--not bad!
X, y = get_data_custom('data-1_train.csv', 2, 3, False)
y = one_hot_encode(y)


kf = KFold(n_splits=10)
kf.get_n_splits(X)
accuracies = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
       
    ffnn = Sequential()
    ffnn.add(Dense(8, input_dim = len(X_train[0]), activation = 'relu', kernel_regularizer = l1(0)))
    #add a second hidden layer, usually fewer and fewer nodes per hidden layer, this is such a small example it's way overdone
    ffnn.add(Dense(4, activation = 'relu', kernel_regularizer = l1(0)))
    #softmax used in output layer, output layer must match number or categories, for whatever reason we are using 0, 1, 2 and 3
    ffnn.add(Dense(3, activation = 'softmax', kernel_regularizer = l1(0)))
    ffnn.compile(optimizer = SGD(lr = 0.1),
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    
    
    print("Training on " + str(len(X[0])) + " features...")
    ffnn.fit(x = X_train, y = y_train, epochs = 50, batch_size = 50)
    print("Done training...")
    accuracies.append(ffnn.evaluate(X_test, y_test)[1])

for acc in accuracies:
    print(acc)
    
print("Average accuracy: " + str(np.average(accuracies)))