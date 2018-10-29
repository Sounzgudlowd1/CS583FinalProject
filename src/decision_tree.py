# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 09:53:20 2018

@author: Erik
"""

from get_data import get_data, one_hot_encode
from sklearn.model_selection import train_test_split
from sklearn import tree

print("Decision Tree:\n")


#you can replace this with whatver data getting method you want to try:
        #file name, max gram length, min occurances of gram
X, y = get_data('data-1_train.csv', 3, 3)





y =  one_hot_encode(y) #seems optional but leave it for now
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

dec_tree = tree.DecisionTreeClassifier()
print("Training...")
dec_tree.fit(X_train, y_train)
print("Done training...")

train_score = dec_tree.score(X_train, y_train)
test_score = dec_tree.score(X_test, y_test)
print("Train accuracy: " + str(train_score))
print("Test accuracy: " + str(test_score))