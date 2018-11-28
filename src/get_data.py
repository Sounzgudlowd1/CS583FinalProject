# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 15:36:35 2018

@author: Erik
"""

from stop_words import stop_words as cust_stop_words
import string
from nltk import ngrams
import nltk
import numpy as np
from keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer

def get_positive_words():
    #taken from https://www.the-benefits-of-positive-thinking.com/list-of-positive-words.html
    file = open("../data/PositiveWords.csv", 'r')
    positive_words = []
    for word in file:
        #strip off new line character at the end
        positive_words.append(word[:-1].lower())
    return positive_words

def get_negative_words():
    #https://www.enchantedlearning.com/wordlist/negativewords.shtml
    file = open("../data/NegativeWords.csv", 'r')
    negative_words = []
    for word in file:
        if word != "":
            negative_words.append(word[:-1].lower())
    return negative_words


def get_raw_data(file_name):
    file = None
    try:
        file = open("../data/" + file_name, 'r')
    except:
        print("Unable to open " + file_name)
        return None
    
    #dictionary of classes, the key is the sentence, the element is the class
    sentences = []
    aspects = []
    classes = []
    
    for i, line in enumerate(file):
        if i == 0:
            continue
        sid, sent, term, term_location, output_class = line.split(",")
        sent = sent.lower()
        term = term.lower()
        #comes with new line character, so get rid of that
        if "\n" in output_class:
            output_class = output_class.replace("\n", "")
        sentences.append(sent)
        aspects.append(term)
        classes.append(output_class)
    
    return sentences, aspects, classes

def _remove_stop_words(sentence):
    for sw in cust_stop_words:
        while sw in sentence:
            sentence.remove(sw)
    return sentence

def normalize(sent, remove_stop_words = False):
    sent = sent.replace("[comma]", "")
    #get rid of punctuation
    translator = str.maketrans('', '', string.punctuation)
    sent = sent.translate(translator)
    sent = sent.lower()
    sent = nltk.word_tokenize(sent)
    
    #remove stop words
    if(remove_stop_words):
        sent = _remove_stop_words(sent)

            
    return sent
    
    
def get_grams_up_to(sentence, n):
    sentence_grams = []
    #go until you've hit the number of grams or the length of the sentence, whicheve is first
    for i in range(1, min(len(sentence), n ) + 1):
        #bigrams, then trigrams then....
        sentence_grams.extend(list(ngrams(sentence, i)))
    return sentence_grams

    

def get_ngram_counts(sentences, n):
    gram_count_dict = {}
    #do counting of words
    for sentence in sentences:
        sentence = normalize(sentence)
        #unigrams are manditory
        sentence_grams = get_grams_up_to(sentence, n)
            
    
        for gram in sentence_grams:
            str_gram = str(gram)
            if str_gram in gram_count_dict:
                gram_count_dict[str_gram] += 1
            else:
                gram_count_dict[str_gram] = 1
    gram_counts = []
    
    #now append counts to a list
    for gram in gram_count_dict:
        gram_counts.append((gram, gram_count_dict[gram]))
    
    #sort the list and return it
    gram_counts = sorted(gram_counts, key = lambda x : x[1], reverse = True)
    return gram_counts
    
#get words with counts above X.
def grams_with_count_at_least(grams, count):
    return [gram for gram in grams if gram[1] >= count]

def position_dictionary(grams):
    pos_dict = {}
    word_dict = {}
    for i, gram in enumerate(grams):
        word_dict[gram[0]] = i
        pos_dict[i] = gram[0]
    return pos_dict, word_dict

def retrieve_gram(gram):
    phrase = ""
    for word in gram:
        phrase += word + " "
    phrase = phrase[:-1]
    return phrase    
    
def distance_function(distance):
    return 1/(distance + 1)

def vectorize(sentence, aspect, word_dict, grams_up_to, remove_stop_words):
    #establish a vecotr of zeros, one for each n gram in the vocabulary
    sent_vect = np.zeros(len(word_dict))
    
    #same processing as grams to remove stop words and stuff, make sure it consistent
    sentence = normalize(sentence, remove_stop_words)
    sentence = retrieve_gram(sentence)
    
    #split sentence to left and right of aspect
    left_of_aspect = sentence[0: sentence.find(aspect)]
    right_of_aspect = sentence[sentence.find(aspect) + len(aspect) + 1:]
    
    #now normalize
    left_of_aspect_n = normalize(left_of_aspect)
    right_of_aspect_n = normalize(right_of_aspect)
    
    left_of_aspect = retrieve_gram(left_of_aspect_n)
    right_of_aspect = retrieve_gram(right_of_aspect_n)
    
    #and extract the grams
    left_grams = get_grams_up_to(left_of_aspect_n, grams_up_to)
    right_grams = get_grams_up_to(right_of_aspect_n, grams_up_to)
    
    #find position relative to aspect
    for gram in left_grams:
        if str(gram) not in word_dict:
            continue
        phrase = retrieve_gram(gram)
        right_chunk = left_of_aspect[left_of_aspect.rfind(phrase) + len(phrase) + 1:]
        distance = len(right_chunk.split())
        value = distance_function(distance)
        if value > sent_vect[word_dict[str(gram)]]:
            sent_vect[word_dict[str(gram)]] = value
    
    for gram in right_grams:
        if str(gram) not in word_dict:
            continue
        phrase = retrieve_gram(gram)
        left_chunk = right_of_aspect[0: right_of_aspect.find(phrase)]
        distance = len(left_chunk.split())
        value = distance_function(distance)
        if value > sent_vect[word_dict[str(gram)]]:
            sent_vect[word_dict[str(gram)]] = value
    return sent_vect
            
    
def get_data_custom(file_name, max_gram_length, min_gram_occurances, remove_stop_words = False):
    sentences, aspects, classes = get_raw_data(file_name) #list of sentence, aspect,
    grams = get_ngram_counts(sentences, max_gram_length)
    grams = grams_with_count_at_least(grams, min_gram_occurances)
    pos_dict, word_dict = position_dictionary(grams)
    X = []
    y = []
    for i in range(len(sentences)):
        X.append(vectorize(sentences[i], aspects[i], word_dict, max_gram_length, remove_stop_words))
        y.append(classes[i])
    
    X = np.array(X)
    y = np.array(y)
    return X, y

def get_data_tfidf(file_name):
    sentences, aspects, classes = get_raw_data(file_name)
    vectorizer = TfidfVectorizer(ngram_range = (1, 2), stop_words = cust_stop_words)
    X = vectorizer.fit_transform(sentences)
    return X.toarray(), np.array(classes)
    
    
def one_hot_encode(y):
    y = y.astype(int)
    y += 1 #set to 0, 1, 2 instead of -1, 0, 1
    return to_categorical(y)
    
def unencode(y):
    y = np.argmax(y, axis = 1)
    y -= 1
    return y.astype(str)

