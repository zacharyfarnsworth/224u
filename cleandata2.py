from collections import Counter
import numpy as np
import os
import pandas as pd
import random
from pandas import ExcelWriter
from pandas import ExcelFile
import csv
from nltk.tokenize import RegexpTokenizer
from datetime import datetime
from datetime import timedelta 
from pandas_datareader import data
from yahoofinancials import YahooFinancials
import re
import pickle


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

	with open('LoughranMcDonald_MasterDictionary_2018.csv',encoding='utf8') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			if line_count == 0:
				print(f'Column names are {", ".join(row)}')
				line_count += 1
			else:
				features = np.zeros(numfeatures)
				for i in range(7,17):
					val = float(row[i])
					if val > 100:
						val = 1.0
					features[i-7] = val
				masterdict[row[0]] = features
				line_count += 1
		#print(masterdict)
	

	negated = False
	prev = None

	sentiments = []

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
				if (row[3],date) not in content:
					if prev != None:
						sentiments.append(prev)
					content[(row[3],date)] = np.zeros(10)
					prev = np.zeros(10)
				words = regex.sub('', row[13])
				words = tokenizer.tokenize(words)
				for i in range(len(words)):
					word = words[i].upper()
					if word in masterdict:
						feats = masterdict[word]
						if negated:
							temp = feats[0]
							feats[0] = feats[1]
							feats[1] = temp
							negated = False
							#print('negating!')
						if word in negations:
							negated = True
						prev += feats
				line_count += 1

		sentiments.append(prev)

		print('here')
		print(len(sentiments))

	with open('statements.csv',encoding='utf8') as csv_file:
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
				if (row[3],date) not in content:
					content[(row[3],date)] = np.zeros(10)
				words = regex.sub('', row[13])
				words = tokenizer.tokenize(words)
				for i in range(len(words)):
					word = words[i].upper()
					if word in masterdict:
						feats = masterdict[word]
						if negated:
							temp = feats[0]
							feats[0] = feats[1]
							feats[1] = temp
							negated = False
							#print('negating!')
						if word in negations:
							negated = True
						content[(row[3],date)] += feats
				line_count += 1

		print('here')

		k = 0

		for key, value in content.items():

			try:

				if k % 100 == 0:
					print(k)
				k += 1
				nextday = key[1] + timedelta(days=1)
				nextnextday = nextday + timedelta(days=5)
				start_date = str(nextday.date())
				end_date = str(nextnextday.date())
				panel_data = data.DataReader(key[0], 'iex', nextday, nextnextday)

				start = panel_data['open'][0]

				nextday = key[1] + timedelta(days=30)
				nextnextday = nextday + timedelta(days=5)
				start_date = str(nextday.date())
				end_date = str(nextnextday.date())
				panel_data = data.DataReader(key[0], 'iex', nextday, nextnextday)

				end = panel_data['close'][0]
				
				label = 1 if start < end else 0

				nextdaydict[key] = label

				x.append(value)
				y.append(label)

			except Exception:
				continue

		x = np.array(x)
		y = np.array(y)

		print(x.shape)
		print(y.shape)

		#print(x)

		print('saving')

		np.savetxt('negated_features.txt', x, fmt='%f')
		np.savetxt('negated_labels.txt', y, fmt='%d')

		with open('nextday30dict.pickle', 'wb') as handle:
			pickle.dump(nextdaydict, handle, protocol=pickle.HIGHEST_PROTOCOL)
		#b = np.loadtxt('traindata.txt', dtype=float)



if __name__ == "__main__":
   main()