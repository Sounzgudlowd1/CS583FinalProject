# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 09:53:20 2018

@author: Erik
"""

from get_data import get_data_custom, one_hot_encode
from sklearn import tree
from sklearn.model_selection import KFold
from Score import Score, average_scores




print("Decision Tree:\n")


#you can replace this with whatver data getting method you want to try:
        #file name, max gram length, min occurances of gram
X, y = get_data_custom('data-1_train.csv', 1, 1)

kf = KFold(n_splits=10)
kf.get_n_splits(X)
test_scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    dec_tree = tree.DecisionTreeClassifier()
    dec_tree.fit(X_train, y_train)
    
    y_pred = dec_tree.predict(X_test)
    test_scores.append(Score(y_test, y_pred))
    
    
average_score = average_scores(test_scores)

for score in test_scores:
    print(score.accuracy)
    
print("average accuracy: " + str(average_score.accuracy))
