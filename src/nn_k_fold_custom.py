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
from get_data import get_data_custom, one_hot_encode, unencode
from Score import Score, average_scores


np.random.seed(7)
#file name, max gram length, min occurances of gram, remove stop words(T/F)
#X, y = get_data_custom('data-1_train.csv', 2, 0, False) results in roughly 72% accuracy--not bad!
X, y = get_data_custom('data-1_train.csv', 2, 0, False)
y_encode = one_hot_encode(y)


kf = KFold(n_splits=10)
kf.get_n_splits(X)
scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_encode[train_index], y[test_index]
       
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
    y_pred = ffnn.predict(X_test)
    y_pred = unencode(y_pred)
    scores.append(Score(y_test, y_pred))

for score in scores:
    print(score.accuracy)

avg_score = average_scores(scores)
print("Average accuracy: " + str(avg_score.accuracy))