#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
##
## A supervised Classifier that predicts whether a given piece of text(SMS) is
## SPAM or NOT SPAM(HAM)
##
## Classifier Used: Support Vector Machine
## ----------------
##
## Dataset used:
## --------------
## https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
##
'''

import matplotlib.pyplot as plt
import regex as re
import pandas
import csv
import numpy as np
import sys
import string
import unicodedata
import cPickle
import time

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
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
	#loads training data
	data = load_dataset()
		
	### splitting training and testing data
	message_train, message_test, label_train, label_test = train_test_split(data['message'],data['label'],test_size=0.1)

	try:
		try:
			## checks for exisiting classifier
			spam_detector_SVM = cPickle.load(open("spam_detector_classifier.pkl","r"))
			opt = int(raw_input('Classifier found. Continue(0) or retrain?(1): '))
			
			if opt == 0:
				pass

			elif opt == 1:
				spam_detector_SVM = build_classifier(message_train, message_test, label_train, label_test)

			else:
				print 'Invalid option.'
				sys.exit(0)

		except:
			## creates a new classifier
			print 'Existing Classifier not found. Building Classifier from dataset.'
			spam_detector_SVM = build_classifier(message_train, message_test, label_train, label_test)
			
	
		test_arr = [ "FREE FREE FREE. Dial Now!", "miss you mom", "Consult now to win prizes", 
				"Love Pizza, Win chances to eat unlimited pizza", "this shit sucks main"]

		
		
		verbose_predict_sms(spam_detector_SVM,test_arr)
		#print spam_detector_SVM.best_estimator_
		
		classifier_info(spam_detector_SVM, message_test, label_test)

		# Save the classifier in pickle file for loading instant spam_detector_classifier
		with open("spam_detector_classifier.pkl","wb") as classifier:
			cPickle.dump(spam_detector_SVM,classifier)

	except Exception as e:
		print e


def predict_sms(estimator, array):
	print '\nPredictions:\n=============\n'
	predictions = estimator.predict(array)
	for i in range(len(array)):
		#print predictions[i] , "  :  ", array[i]
		print '{:<6}'.format(predictions[i]), ": ", array[i]
	print

def verbose_predict_sms(estimator, array):
	''' give detailed information about the prediction '''
	print '\nPredictions:\n=============\n'
	#print '    %s              %s     ' % ("HAM","SPAM")
	predictions = estimator.predict(array)
	for i in range(len(array)):
		print
		print '{:<6}'.format(predictions[i]), ": ", array[i]
		print
		print 'Probabilities (HAM / SPAM)'
		print estimator.predict_proba(array[i])[0]
		print ' -------------------------------- '


def classifier_info(estimator, message_test, label_test):
	''' defines information about the classifier '''

	print '\nclassifier used:\n----------------\n'
	print estimator.grid_scores_

	print "Accuracy: ", accuracy_score(label_test, estimator.predict(message_test))
	print "\nConfusion Matrix:\n-----------------\n"
	print confusion_matrix(label_test, estimator.predict(message_test))
	print "\nclassification report:\n----------------------"
	print classification_report(label_test, estimator.predict(message_test))


def build_classifier(message_train, message_test, label_train, label_test):

	###################################
	### Naive Bayes implementation ####
	###################################

	# print label_test
	# #Pipeline_NB for above steps
	# pipeline_NB = Pipeline([
	# 	("vector",CountVectorizer(analyzer=tokenizer)),
	# 	("tfidf",TfidfTransformer()),
	# 	("classifier", MultinomialNB())
	# ])

	# scores = cross_val_score(pipeline_NB,message_train,label_train,cv=10,scoring='accuracy',n_jobs=-1)
	# #print scores

	# params_NB = { 'tfidf__use_idf' : (True, False)}
	# tuned_clf_NB = GridSearchCV(pipeline_NB,params_NB, n_jobs=-1,scoring='accuracy', cv= StratifiedKFold(label_train,n_folds=5))
	
	# spam_detector_NB = tuned_clf_NB.fit(message_train,label_train)

	# return spam_detector_NB


	######################################
	######################################
	

	############################
	# ### SVM implementation ###
	# ##########################

	pipeline_SVM = Pipeline([
			("vector", CountVectorizer(analyzer=tokenizer)),
			("tfidf", TfidfTransformer()),
			("classifier", SVC(probability=True	)),
		]) 

	params_SVM = [{ "classifier__C": [1,10,50,100], "classifier__kernel":["linear","rbf"], "classifier__gamma":[.001,.0001]}]

	tuned_clf_SVM = GridSearchCV(pipeline_SVM, params_SVM, refit="True", n_jobs=-1,
							scoring="accuracy",cv=StratifiedKFold(label_train,n_folds=5))

	spam_detector_SVM = tuned_clf_SVM.fit(message_train,label_train)

	return spam_detector_SVM
	


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """  Generate a simple plot of the test and traning learning curve. """

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


def tokenizer(message):
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