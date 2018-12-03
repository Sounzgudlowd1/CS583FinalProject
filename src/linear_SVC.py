# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 16:28:11 2018

@author: Erik
"""
from  sklearn.svm import LinearSVC
from get_data import get_data_tfidf, get_data_custom
from sklearn.model_selection import KFold
from Score import Score, average_scores

X, y = get_data_tfidf('data-2_train.csv')
#X, y = get_data_custom('data-2_train.csv', 2, 1)  # try data1 and data2 and custom v tfidf

kf = KFold(n_splits=10)
kf.get_n_splits(X)
scores = []


i = 0
for train_index, test_index in kf.split(X):
    i += 1
    print("iteration: " + str(i))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    scores.append(Score(y_test, y_pred))

avg_score = average_scores(scores)
print("Average test accuracy " + str(avg_score.accuracy))
