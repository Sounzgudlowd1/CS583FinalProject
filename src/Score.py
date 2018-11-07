# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 14:47:09 2018

@author: Erik
"""
import numpy as np
class Score:
    def __init__(self, y_true = None, y_pred = None):
        if y_true is not None:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            self.set_accuracy(y_true, y_pred)
            self.precision_positive = self.precision(y_true, y_pred, "1")
            self.recall_positive = self.recall(y_true, y_pred, "1")
            self.f1_positive = self.f1_score(y_true, y_pred, "1")
            
            self.precision_neutral = self.precision(y_true, y_pred, "0")
            self.recall_neutral = self.recall(y_true, y_pred, "0")
            self.f1_neutral = self.f1_score(y_true, y_pred, "0")
            
            self.precision_negative = self.precision(y_true, y_pred, "-1")
            self.recall_negative = self.recall(y_true, y_pred, "-1")
            self.f1_negative = self.f1_score(y_true, y_pred, "-1")
        else:
            self.accuracy = 0
            self.precision_positive = 0
            self.recall_positive = 0
            self.f1_positive = 0
            
            self.precision_neutral = 0
            self.recall_neutral = 0
            self.f1_neutral = 0
            
            self.precision_negative = 0
            self.recall_negative = 0
            self.f1_negative = 0
        
    def set_accuracy(self, y_true, y_pred):
        self.accuracy = np.sum(y_true == y_pred) / len(y_pred)
        
    def precision(self, y_true, y_pred, label):
        #correctly predicted as true/#predicted as true
        predicted_true_count = 0
        correctly_predicted_true_count = 0
        for i in range(len(y_true)):
            if y_pred[i] == label:
                predicted_true_count += 1
            
            if y_pred[i] == label and y_true[i] == y_pred[i]:
                correctly_predicted_true_count += 1
        if predicted_true_count == 0:
            return 0 #don't know about this, if I made no predictions for this label then what is the precision?
        else:
            return correctly_predicted_true_count / predicted_true_count
    
    def recall(self, y_true, y_pred, label):
        actual_true_count = 0
        correctly_predicted_true_count = 0
        for i in range(len(y_true)):
            if y_true[i] == label:
                actual_true_count += 1
            
            if y_pred[i] == label and y_true[i] == y_pred[i]:
                correctly_predicted_true_count += 1
        if actual_true_count == 0:
            return 0
        else:
            return correctly_predicted_true_count / actual_true_count
    
    def f1_score(self, y_true, y_pred, label):
        prec = self.precision(y_true, y_pred, label)
        rec = self.recall(y_true, y_pred, label)
        if prec + rec == 0:
            return 0
        else:
            return 2 * prec * rec /( prec + rec)

def average_scores(scores):
    avg_score = Score()
    
    #this may be laziness but rather than doing (acc1 + acc2 + acc3 + ... ) /(count)
    # I'm doing (acc1/count + acc2/count + acc3/count)
    for score in scores:
        avg_score.accuracy += score.accuracy / len(scores)
        
        avg_score.precision_positive += score.precision_positive /len(scores)
        avg_score.recall_positive += score.recall_positive /len(scores)
        avg_score.f1_positive += score.f1_positive / len(scores)
        
        avg_score.precision_neutral += score.precision_neutral / len(scores)
        avg_score.recall_neutral += score.recall_neutral / len(scores)
        avg_score.f1_neutral += score.f1_neutral / len(scores)
        
        avg_score.precision_negative += score.precision_negative /len(scores)
        avg_score.recall_negative += score.recall_negative / len(scores)
        avg_score.f1_negative += score.f1_negative / len(scores)
    return avg_score
        
    

                