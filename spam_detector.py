#!/usr/bin/python
# -*- coding: utf-8 -*-

'''

## A supervised Classifier that predicts whether a given piece of text(SMS) is
## SPAM or NOT SPAM(HAM)

## Dataset used:
## --------------
## https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

'''

import matplotlib.pyplot as plt
import regex as re
import pandas
import csv
import numpy as np
import sys
import string
import unicodedata

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.learning_curve import learning_curve



###################
### GLOBAL VARS ###
###################

# creating dict for unicode translation(removal of punctuation)
# reference - http://www.unicode.org/reports/tr44/tr44-4.html#General_Category_Values
table = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(unichr(i)).startswith('P'))

stop_words = stopwords.words("english")

#####################



def main():
	data = load_dataset()
	clf_NB = MultinomialNB()
	
	### splitting training and testing data
	message_train, message_test, label_train, label_test = train_test_split(data['message'],data['label'],test_size=0.1)


	#Pipeline for above steps
	pipeline = Pipeline([
		("vector",CountVectorizer(analyzer=tokenize)),
		("tfidf",TfidfTransformer()),
		("classifier", MultinomialNB())
	])

	scores = cross_val_score(pipeline,message_train,label_train,cv=10,scoring='accuracy',n_jobs=-1)
	#print scores

	params = { 'tfidf__use_idf' : (True, False)}
	tuned_clf_NB = GridSearchCV(pipeline,params, n_jobs=-1,scoring='accuracy', cv= StratifiedKFold(label_train,n_folds=5))
	print len(label_train)
	spam_detector = tuned_clf_NB.fit(message_train,label_train)

	
	test_arr = [ "FREE FREE FREE. Dial Now!", "miss you mom", "Consult now to win prizes", "Love Pizza, Win chances to eat unlimited pizza",
		"this shit sucks main"]

	print spam_detector.predict(test_arr)
	for i in test_arr:
		print spam_detector.predict_proba([i])[0]

	# print spam_detector.best_estimator_
	# print spam_detector.grid_scores_
	# plot = plot_learning_curve(pipeline,"",message_train,label_train,cv=10)
	# plt.show()


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def tokenize(message):
	''' Tokenize/Lemmatize the words'''
	message = unicode(message,'utf-8').lower()
	message = remove_punctuation_unicode(message)
	words = [ word for word in word_tokenize(message) if word not in stop_words ]
	WNL = WordNetLemmatizer()
	return [ WNL.lemmatize(word) for word in words ]

def remove_punctuation_unicode(string):
	return string.translate(table)

def load_dataset():
	try:
		data = pandas.read_csv("spam_dataset", sep="\t", names=["label","message"])
		data["length"] = data["message"].map(lambda sms: len(sms))
		return data

	except Exception as e:
		print e
		print 'Error: Dataset not loaded.'
		sys.exit(1)
		


if __name__ == '__main__':
	main()