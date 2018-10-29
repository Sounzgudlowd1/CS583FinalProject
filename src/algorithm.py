# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 15:30:36 2018

@author: Erik
"""
from get_data import get_data, get_positive_words, get_negative_words
from copy import deepcopy  

def evaluate(data, sentence, positive_words, negative_words):
    predictions = []
    actual = []
    for elt in data[sentence].aspects:
    
        before_aspect = sentence[:sentence.find(elt.aspect_term)]
        after_aspect = sentence[sentence.rfind(elt.aspect_term) + len(elt.aspect_term):]
        before_aspect = before_aspect.split(" ")
        after_aspect = after_aspect.split(" ")
        
        after_aspect_index = 0
        before_aspect_index = len(before_aspect) - 1
        while True:
            if after_aspect_index < len(after_aspect) and after_aspect[after_aspect_index] in positive_words:
                predictions.append(1)
                break

            if after_aspect_index < len(after_aspect) and after_aspect[after_aspect_index] in negative_words:
                predictions.append(-1)
                break

            if before_aspect_index >= 0 and before_aspect[before_aspect_index] in positive_words:
                predictions.append(1)
                break
            
            if before_aspect_index >= 0 and before_aspect[before_aspect_index] in negative_words:
                predictions.append(-1)
                break
            
            after_aspect_index += 1
            before_aspect_index -= 1
            if after_aspect_index >  len(after_aspect) and before_aspect_index < 0:
                predictions.append(0)
                break
        actual.append(elt.output_class)
    return actual, predictions


def evaluate_data_set(d, pw, nw):
    results = []
    for i, sent in enumerate(d):
        res = evaluate(d, sent, pw, nw)
        results.append(res)
    return results

def get_stats(results):
    sentence_matches = 0
    aspect_matches = 0
    sentence_count = 0
    aspect_count = 0
    file = open("output.csv", 'w')

    for res in results:
        file.write(str(res) + "\n")
        
        if res[0] == res[1]:
            sentence_matches += 1
        sentence_count += 1
            
        for i in range(len(res[0])):
            if res[0][i] == res[1][i]:
                aspect_matches += 1
            aspect_count += 1
    file.close()
    return aspect_matches/aspect_count, sentence_matches/sentence_count

def optimize(d, v, pw = [], nw = []):
    results = evaluate_data_set(d, pw, nw)
    aspect_accuracy, sentence_accuracy = get_stats(results)
    best_aspect_accuracy = aspect_accuracy
    best_pw = pw
    best_nw = nw
    print("initial accuracy: " + str(best_aspect_accuracy))
    while(True):
        #best accuracy found this iteration
        iter_best_aspect_accuracy = best_aspect_accuracy
        for word in v:
            print("Working on: " + word)
            temp_pw = deepcopy(best_pw)
            temp_pw.append(word)
            new_results = evaluate_data_set(d, temp_pw, best_nw)
            new_aspect_accuracy, new_sentence_accuracy = get_stats(new_results)
            if(new_aspect_accuracy > best_aspect_accuracy):
                print(new_aspect_accuracy)
                best_aspect_accuracy = new_aspect_accuracy
                best_pw = temp_pw
            
            temp_nw = deepcopy(best_nw)
            temp_nw.append(word)
            new_results = evaluate_data_set(d, best_pw, temp_nw)
            new_aspect_accuracy, new_sent_accuracy = get_stats(new_results)
            if(new_aspect_accuracy > best_aspect_accuracy):
                print(new_aspect_accuracy)
                best_aspect_accuracy = new_aspect_accuracy
                best_nw = temp_nw
        break
    return best_pw, best_nw
        
      
        
d, v = get_data("data-1_train.csv")
d_train = {}
d_test = {}

for i, sent in enumerate(d):
    if i < 1000:
        d_train[sent] = d[sent]
    else:
        d_test[sent] = d[sent]
    
    
    
pw = get_positive_words()
nw = get_negative_words()

best_pw, best_nw = optimize(d_train, v, pw, nw)

train_results = evaluate_data_set(d_train, best_pw, best_nw)
train_aspect_accuracy, train_sentence_accuracy = get_stats(train_results)

test_results = evaluate_data_set(d_test, best_pw, best_nw)
test_aspect_accuracy, test_sentence_accuracy = get_stats(test_results)

print("Stats on train: " + str(train_aspect_accuracy))
print("Stats on test: " + str(test_aspect_accuracy))
    