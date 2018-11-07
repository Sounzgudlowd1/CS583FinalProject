# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 19:45:01 2018

@author: Erik
"""

from get_data import get_data_custom, get_data_tfidf, one_hot_encode, unencode
from sklearn import tree
from sklearn.model_selection import KFold
from Score import Score, average_scores
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from keras.optimizers import SGD
from keras.regularizers import l1, l2, l1_l2

DATA_SETS = ['data-1_train.csv', 'data-2_train.csv']
ALGOS = ['nn', 'nb', 'dt', 'rf']
PRE_PROCS = ['tfidf', 'cust']

file = open('test.csv', 'w')
file.write("test")


for ds in DATA_SETS:
    for alg in ALGOS:
        for proc in PRE_PROCS:
            if proc == 'tfidf':
                X, y = get_data_tfidf(ds)
            else:
                X, y = get_data_custom(ds, 2, 0, False)
            
            y_encode = one_hot_encode(y)


            kf = KFold(n_splits=10)
            kf.get_n_splits(X)
            scores = []
            
            print("Working on: " + ds + " " + alg + " " + proc)
            i = 1
            for train_index, test_index in kf.split(X):
                print(i)
                i += 1
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y_encode[train_index], y[test_index]
                
                if alg == 'nn':
                    clf = Sequential()
                    clf.add(Dense(8, input_dim = len(X_train[0]), activation = 'relu', kernel_regularizer = l1(0)))
                    #add a second hidden layer, usually fewer and fewer nodes per hidden layer, this is such a small example it's way overdone
                    clf.add(Dense(4, activation = 'relu', kernel_regularizer = l1(0)))
                    #softmax used in output layer, output layer must match number or categories, for whatever reason we are using 0, 1, 2 and 3
                    clf.add(Dense(3, activation = 'softmax', kernel_regularizer = l1(0)))
                    clf.compile(optimizer = SGD(lr = 0.1),
                                  loss = 'categorical_crossentropy',
                                  metrics = ['accuracy'])
                    clf.fit(x = X_train, y = y_train, epochs = 50, batch_size = 50)
                    
                elif alg == 'dt':
                    clf = tree.DecisionTreeClassifier()
                    clf.fit(X_train, y_train)
                
                elif alg == 'rf':
                    clf = RandomForestClassifier(n_estimators = 100)
                    clf.fit(X_train, y_train)
                
                elif alg == 'nb':
                    #need original categorical for naive bayes
                    y_train, y_test = y[train_index], y[test_index]
                    clf = GaussianNB()
                    clf.fit(X_train, y_train)
                
                else:
                    print("unkown classifier " + alg)
                
                y_pred = clf.predict(X_test)
                if alg != 'nb':
                    y_pred = unencode(y_pred)
                scores.append(Score(y_test, y_pred))
            avg_score = average_scores(scores)
            file.write(ds+ "," + alg + "," + proc + "," + 
                       str(avg_score.accuracy) + ", " +
                       str(avg_score.f1_positive) + "," +
                       str(avg_score.precision_positive)+ "," +
                       str(avg_score.recall_positive) + "," +
                       
                       str(avg_score.f1_neutral) + "," +
                       str(avg_score.precision_neutral) + "," +
                       str(avg_score.recall_neutral) + "," +
                       
                       str(avg_score.f1_negative) + "," +
                       str(avg_score.precision_negative) + "," +
                       str(avg_score.recall_negative) + "\n"
                       )
            print("accuracy")
            print(avg_score.accuracy)
                
                
file.close()                





