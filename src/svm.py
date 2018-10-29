# -*- coding: utf-8 -*-
"""
THIS ALWAYS OUTPUTS THE MAJORITY CLASS!!!  Something is broken about the C and gamma values most likely but I can't find a good
combination so far.
"""


from get_data import get_data
from sklearn.model_selection import train_test_split
from sklearn import svm

print("SVM:\n")

#file name, max grams, in occurances of gram
X, y = get_data('data-1_train.csv', 3, 3)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
X_train = X[0:4]
X_test = X[4: 6]
y_train = y[0: 4]
y_test = y[4:6]

sup_vec = svm.SVC(C = 100, kernel = 'rbf', gamma = 1e-20)
print("Training...")
sup_vec.fit(X_train, y_train)
print("Done training...")

train_score = sup_vec.score(X_train, y_train)
test_score = sup_vec.score(X_test, y_test)
print("Train accuracy: " + str(train_score))
print("Test accuracy: " + str(test_score))