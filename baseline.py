from collections import Counter
import numpy as np
import os
import pandas as pd
import random
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

	x = np.loadtxt('baseline_features.txt', dtype=float)
	y = np.loadtxt('baseline_labels.txt', dtype=float)

	row_sums = x.sum(axis=1)
	x_norm = np.nan_to_num(x / row_sums[:, np.newaxis])
	np.savetxt('test.txt', x_norm, fmt='%f')
	x = x_norm

	x_train, x_test, y_train, y_test = train_test_split(
		x, y, test_size=0.2)

	logisticRegr = LogisticRegression()
	logisticRegr.fit(x_train, y_train)

	score = logisticRegr.score(x_test, y_test)
	print(score)

	# clf = RandomForestClassifier(n_estimators=100)
	# clf.fit(x_train, y_train)  

	# score = clf.score(x_test, y_test)
	# print(score)

	y_pred = logisticRegr.predict(x_test)

	print(np.sum(y_pred))
	print(y_pred.shape)

	print(np.sum(y_test))
	print(y_test.shape)

	print(precision_score(y_test, y_pred, average='macro'))
	print(recall_score(y_test, y_pred, average='macro'))
	print(f1_score(y_test, y_pred, average='macro'))

	# gnb = GaussianNB()
	# gnb.fit(x_train, y_train)

	# score = gnb.score(x_test, y_test)
	# print(score)

	# clf = linear_model.SGDClassifier(max_iter=1000000, tol=1e-10)
	# clf.fit(x_train, y_train)
	# print(clf.score(x_test,y_test))

	# nn = MLPClassifier()
	# nn.fit(x_train, y_train)
	# print(nn.score(x_test,y_test))



if __name__ == "__main__":
   main()