# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 17:39:00 2018

@author: Erik
"""

from sklearn.naive_bayes import MultinomialNB
from get_data import get_data_tfidf, get_data_custom
from Score import Score, average_scores
from sklearn.model_selection import KFold

X, y = get_data_custom("data-2_train.csv", 3, 2)


kf = KFold(n_splits=10)
kf.get_n_splits(X)
test_scores = []
train_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    gnb = MultinomialNB()
    gnb.fit(X_train, y_train)
    
    
    y_pred = gnb.predict(X_test)
    test_scores.append(Score(y_test, y_pred))

    y_pred = gnb.predict(X_train)
    train_scores.append(Score(y_train, y_pred))
    
average_score = average_scores(test_scores)
print("Average test score: " + str(average_score.accuracy))

average_train_score = average_scores(train_scores)
print("Average train score: " + str(average_train_score.accuracy))