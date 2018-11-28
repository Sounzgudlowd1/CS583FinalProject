# -*- coding: utf-8 -*-
"""
THIS ALWAYS OUTPUTS THE MAJORITY CLASS!!!  Something is broken about the C and gamma values most likely but I can't find a good
combination so far.
"""


from get_data import get_data_custom
from sklearn.model_selection import train_test_split
from sklearn import svm

print("SVM:\n")

#file name, max grams, in occurances of gram
X, y = get_data_custom('data-1_train.csv', 2, 2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
#X_train = X[0:4]
#X_test = X[4: 6]
#y_train = y[0: 4]
#y_test = y[4:6]

'''This is what really needs work.  Right now it just outputs the majority class.  We need to tune the hyper
parameters C and gamma.  Please refer to the documentation for their definitions.

The problem is that currently the modl just outputs the majority label ~50% accuracy.  Your task is to find a way to improve it.

A crude attempt at hyper parameter tuning is below
'''
Cs = [0.1, 1, 10, 100, 1000] #iterate through a variety of different C values
gammas = [1e-20, 1e-10, 1e-5, 1e-2]  #iterate through a veriety of different gamma values


for thisC in Cs:
    for thisGamma in gammas:
                
        sup_vec = svm.SVC(C = thisC, gamma = thisGamma) #instantiate an svm with those hyper parameters
        print("Training...")
        sup_vec.fit(X_train, y_train)  # fit it to the data
        print("Done training...")
        predictions = sup_vec.predict(X_test) # get predictions
        print("C: " + str(thisC) + " gamma: " + str(thisGamma) + " num labels " + str(len(set(predictions)))) #report back which paramters are used

train_score = sup_vec.score(X_train, y_train)
test_score = sup_vec.score(X_test, y_test)
print("Train accuracy: " + str(train_score))
print("Test accuracy: " + str(test_score))