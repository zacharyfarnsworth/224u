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
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from scipy.sparse import coo_matrix, hstack

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


	ignoredwords = list(stopwords.words('english'))

	stemmer = SnowballStemmer("english", ignore_stopwords=True)


	texts = []
	labels = []

	with open('nextday30dict.pickle', 'rb') as handle:
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

	# negated = False
	# prev = None

	# sentiments = []

	# with open('qna.csv',encoding='utf8') as csv_file:
	# 	csv_reader = csv.reader(csv_file, delimiter=',')
	# 	line_count = 0
	# 	for row in csv_reader:
	# 		if line_count == 0:
	# 			print(f'Column names are {", ".join(row)}')
	# 			line_count += 1
	# 		else:
	# 			try:
	# 				day = row[10][:row[10].index(',')+6]
	# 			except Exception:
	# 				continue
	# 			try:
	# 				date = datetime.strptime(day, '%B %d, %Y')
	# 			except Exception:
	# 				try:
	# 					date = datetime.strptime(day, '%b %d, %Y')
	# 				except Exception:
	# 					continue
	# 			if (row[3], date) not in nextdaydict:
	# 				continue
	# 			words = regex.sub('', row[13])
	# 			words = tokenizer.tokenize(words)
	# 			totalfeats = np.zeros(10)
	# 			for i in range(len(words)):
	# 				word = words[i].upper()
	# 				if word in masterdict:
	# 					feats = masterdict[word]
	# 					if negated:
	# 						temp = feats[0]
	# 						feats[0] = feats[1]
	# 						feats[1] = temp
	# 						negated = False
	# 						#print('negating!')
	# 					if word in negations:
	# 						negated = True
	# 					totalfeats += feats
	# 			if prev == (row[3],date):
	# 				sentiments[len(sentiments) - 1] += totalfeats
	# 			else:
	# 				sentiments.append(totalfeats)
	# 				prev = (row[3],date)
	# 			line_count += 1

		print('here')

	with open('sentiments30.pickle', 'rb') as fp:
		sentiments = pickle.load(fp)

	#print('dumped')
	
	# prev = None
	# with open('qna.csv',encoding='utf8') as csv_file:
	# 	csv_reader = csv.reader(csv_file, delimiter=',')
	# 	line_count = 0
	# 	for row in csv_reader:
	# 		if line_count % 1000 == 0:
	# 			print(line_count/1000)
	# 		if line_count == 0:
	# 			#print(f'Column names are {", ".join(row)}')
	# 			line_count += 1
	# 		else:
	# 			try:
	# 				day = row[10][:row[10].index(',')+6]
	# 			except Exception:
	# 				continue
	# 			try:
	# 				date = datetime.strptime(day, '%B %d, %Y')
	# 			except Exception:
	# 				try:
	# 					date = datetime.strptime(day, '%b %d, %Y')
	# 				except Exception:
	# 					continue

	# 			if (row[3],date) not in nextdaydict:
	# 				continue

	# 			stems = " ".join([stemmer.stem(x) for x in word_tokenize(row[13])])
	# 			if prev == (row[3],date):
	# 				texts[len(texts)-1] = " ".join([texts[len(texts)-1], stems])
	# 			else:
	# 				texts.append(stems)
	# 				prev = (row[3],date)
	# 				labels.append(nextdaydict[(row[3],date)])
	# 			line_count += 1

	# 	print('here')

	with open('texts30day.pickle', 'rb') as fp:
		texts = pickle.load(fp)

	with open('labels30day.pickle', 'rb') as fp:
		labels = pickle.load(fp)

	#num_feats = [100,1000,10000,30000]
	num_feats = [1000,10000,100000,1000000]
	print(len(texts))
	print(len(labels))
	print(len(sentiments))

	for n in num_feats:

		print('Number of features = ' + str(n))

		vectorizer = CountVectorizer(max_features=n, ngram_range=(2, 2), 
			binary=True, stop_words=ignoredwords)

		#print(texts)

		x_train, x_test, y_train, y_test, s_train, s_test = train_test_split(texts, labels, sentiments, test_size=0.2)

		print('Percent positive = ' + str(np.sum(labels)/len(labels)))

		#print(x_train)

		X_train = vectorizer.fit_transform(x_train)

		X_train = hstack([X_train,np.array(s_train)])

		#X_train = np.append(X_train.data, np.array(s_train).data, axis=1)

		print('done')

		feature_names = vectorizer.get_feature_names()

		#print(X_train.toarray())
		print(X_train.shape)

		X_test = vectorizer.transform(x_test)

		X_test = hstack([X_test,np.array(s_test)])

		print(X_test.shape)

		#Logistic Regression

		# logisticRegr = LogisticRegression(max_iter=10000)
		# logisticRegr.fit(X_train, y_train)

		# y_pred = logisticRegr.predict(X_test)

		# pos = np.sum(y_pred)/len(y_test)

		# print('percent positive predictions = ' + str(pos))

		# score = logisticRegr.score(X_test, y_test)
		# print(score)
		# print(f1_score(y_test, y_pred, average='macro'))

		# coefs = np.abs(logisticRegr.coef_[0])
		# top10 = np.argpartition(coefs, -10)[-10:]
		# top10_sorted = top10[np.argsort(coefs[top10])]
		# for feat in top10_sorted:
		# 	print(feature_names[feat])

		#Naive Bayes

		mnb = MultinomialNB()
		mnb.fit(X_train, y_train)

		y_pred = mnb.predict(X_test)

		score = mnb.score(X_test, y_test)
		print(score)

		print(precision_score(y_test, y_pred, average='macro'))
		print(recall_score(y_test, y_pred, average='macro'))
		print(f1_score(y_test, y_pred, average='macro'))

		coefs = np.abs(logisticRegr.coef_[0])
		top10 = np.argpartition(coefs, -10)[-10:]
		top10_sorted = top10[np.argsort(coefs[top10])]
		for feat in top10_sorted:
			print(feature_names[feat])

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