# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 10:04:55 2018

@author: Erik
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from get_data import get_data_custom, one_hot_encode
#used to split data
from sklearn.model_selection import train_test_split

#file name, max gram length, min occurances of gram
#for me  get_data('data-1_train.csv', 1, 0) is around 71-73% accuracy on test, which is actually great!
#  and get_data_custom('data-1_train.csv', 3, 3) yeilds around 72 as well but takes much longer
X, y = get_data_custom('data-1_train.csv', 2, 0, False)
y = one_hot_encode(y)


#split as required
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size = 0.2)

ffnn = Sequential()
ffnn.add(Dense(1024, input_dim = len(X_train[0]), activation = 'relu'))
#add a second hidden layer, usually fewer and fewer nodes per hidden layer, this is such a small example it's way overdone
ffnn.add(Dense(512, activation = 'relu'))
#softmax used in output layer, output layer must match number or categories, for whatever reason we are using 0, 1, 2 and 3
ffnn.add(Dense(3, activation = 'softmax'))
ffnn.compile(optimizer = SGD(lr = 0.1),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])


print("Training on " + str(len(X[0])) + " features...")
ffnn.fit(x = X_train, y = y_train, epochs = 50, batch_size = 50, 
          validation_data = [X_validation, y_validation])
print("Done training...")

train_score = ffnn.evaluate(X_train, y_train)
test_score = ffnn.evaluate(X_test, y_test)
print("Train accuracy: " + str(train_score[1]))
print("Test accuracy: " + str(test_score[1]))