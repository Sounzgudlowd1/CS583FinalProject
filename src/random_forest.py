# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 11:29:15 2018

@author: Erik
"""

from sklearn.ensemble import RandomForestClassifier
from get_data import get_data_custom, get_data_tfidf, one_hot_encode
from sklearn.model_selection import train_test_split
print("Random Forest:\n")


X, y = get_data_tfidf('data-1_train.csv')
y =  one_hot_encode(y) #seems optional but leave it for now
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

#1000 trees is better than one?
rfc = RandomForestClassifier(n_estimators = 100)
print("Training...")
rfc.fit(X_train, y_train)
print("Done training...")

train_score = rfc.score(X_train, y_train)
test_score = rfc.score(X_test, y_test)
print("Train accuracy: " + str(train_score))
print("Test accuracy: " + str(test_score))