import numpy as np
import os
import random
import csv
from nltk.tokenize import RegexpTokenizer
from datetime import datetime
from datetime import timedelta 
import re
import pickle
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier


def main():

	content = {}
	tickers = {}

	nextdaydict = {}

	masterdict = {}

	x = []
	y = []

	numfeatures = 10

	tokenizer = RegexpTokenizer(r'\w+')
	regex = re.compile('[^\sa-zA-Z]')

	negations = {'NEVER','NO','NOTHING','NOWHERE','NOONE','NONE','NOT','HAVENT',
		'HASNT','HADNT','CANT','COULDNT','SHOULDNT','WONT','WOULDNT','DONT',
		'DOESNT','DIDNT','ISNT','ARENT','AINT'}


	#ignoredwords = set(stopwords.words('english'))


	texts = []
	labels = []

	with open('nextdaydict.pickle', 'rb') as handle:
		nextdaydict = pickle.load(handle)

	# with open('LoughranMcDonald_MasterDictionary_2018.csv',encoding='utf8') as csv_file:
	# 	csv_reader = csv.reader(csv_file, delimiter=',')
	# 	line_count = 0
	# 	for row in csv_reader:
	# 		if line_count == 0:
	# 			print(f'Column names are {", ".join(row)}')
	# 			line_count += 1
	# 		else:
	# 			features = np.zeros(numfeatures)
	# 			for i in range(7,17):
	# 				val = float(row[i])
	# 				if val > 100:
	# 					val = 1.0
	# 				features[i-7] = val
	# 			masterdict[row[0]] = features
	# 			line_count += 1
		#print(masterdict)
	

	with open('qna.csv',encoding='utf8') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			if line_count == 0:
				print(f'Column names are {", ".join(row)}')
				line_count += 1
			else:
				try:
					day = row[10][:row[10].index(',')+6]
				except Exception:
					continue
				try:
					date = datetime.strptime(day, '%B %d, %Y')
				except Exception:
					try:
						date = datetime.strptime(day, '%b %d, %Y')
					except Exception:
						continue

				if (row[3],date) not in nextdaydict:
					continue
				texts.append(row[13])
				labels.append(nextdaydict[(row[3],date)])
				line_count += 1

		print('here')

	vectorizer = CountVectorizer(max_features=30000, ngram_range=(2, 2))

	#print(texts)

	x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)

	#print(x_train)

	X_train = vectorizer.fit_transform(x_train)

	#print(X_train.toarray())
	print(X_train.shape)

	X_test = vectorizer.transform(x_test)

	print(X_test.shape)

	#Logistic Regression

	logisticRegr = LogisticRegression()
	logisticRegr.fit(X_train, y_train)

	y_pred = logisticRegr.predict(X_test)

	score = logisticRegr.score(X_test, y_test)
	print(score)

	print(precision_score(y_test, y_pred, average='macro'))
	print(recall_score(y_test, y_pred, average='macro'))
	print(f1_score(y_test, y_pred, average='macro'))

	#GaussianNB

	# gnb = GaussianNB()
	# gnb.fit(X_train.toarray(), y_train)

	# y_pred = gnb.predict(X_test)

	# score = gnb.score(X_test.toarray(), y_test)
	# print(score)

	# print(precision_score(y_test, y_pred, average='macro'))
	# print(recall_score(y_test, y_pred, average='macro'))
	# print(f1_score(y_test, y_pred, average='macro'))

	#SGD

	# clf = linear_model.SGDClassifier(max_iter=1000000, tol=1e-10)
	# clf.fit(X_train, y_train)
	# print(clf.score(X_test,y_test))

	# y_pred = clf.predict(X_test)

	# print(precision_score(y_test, y_pred, average='macro'))
	# print(recall_score(y_test, y_pred, average='macro'))
	# print(f1_score(y_test, y_pred, average='macro'))


if __name__ == "__main__":
   main()