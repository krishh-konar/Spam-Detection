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
import pandas
import csv
import numpy as np
import sys

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

stop_words = stopwords.words("english")

def main():
	data = loadDataset()
	tokens = data.message.apply(tokenize)
	

	vector = CountVectorizer(analyzer=tokenize).fit(data['message'])
	print len(vector.vocabulary_)

	msg = vector.transform(data['message'][3])
	print data['message'][3]
	print msg
	# print data.head()
	# print tokens
	#print data.groupby("label").describe()

	# data.hist(bins=50 , column="length", by="label")
	# plt.show()

def tokenize(message):
	msg = unicode(message,"utf-8").lower()
	words = [ word for word in word_tokenize(msg) if word not in stop_words ]
	WNL = WordNetLemmatizer()
	return [ WNL.lemmatize(word) for word in words ]

def loadDataset():
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