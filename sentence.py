# Importing all the necessary files
import numpy as np
import pandas as pd
import glob
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB



# Collecting the location of all labeled articles
filenames = glob.glob('/home/shubham/SentenceCorpus/labeled_articles/*.txt')


dataset = []

# Reading the file and creating the dataset
for file in filenames:
	f = open(file, "r")
	read = f.readlines()
	for line in read:
		if line.startswith("#"):                       # to remove ## abstract ## and ## introduction ## line
			pass
		else:
			dataset.append(line)
	f.close()



# Collecting all the stopwords in a list
file = pd.read_csv('/home/shubham/SentenceCorpus/word_lists/stopwords.txt')
stopwords = []
for word in file:
    if word.endswith('\n'):
        word = word[:-1]
        stopwords.append(word)



# pre-processing of dataset i.e., separating the category and sentence, removing the stopwords, etc
data_category = []
data_sentence = []
for line in dataset:
	if "\t" in line:
		splits = line.split("\t")                         # split if tab is present
		s_category = splits[0]
		sentence = splits[1].lower()
		for sw in stopwords:
			sentence = sentence.replace(sw, "")
		pattern = re.compile("[^\w']")                    # remove any character not present in unicode database
		sentence = pattern.sub(' ', sentence)
		sentence = re.sub(' +', ' ', sentence)
		data_category.append(s_category)
		data_sentence.append(sentence)
	else:
		splits = line.split(" ")
		s_category = splits[0]
		sentence = line[len(s_category)+1:].lower()
		for sw in stopwords:
			sentence = sentence.replace(sw, "")
		pattern = re.compile("[^\w']")
		sentence = pattern.sub(' ', sentence)
		sentence = re.sub(' +', ' ', sentence)
		data_category.append(s_category)
		data_sentence.append(sentence)


data = pd.DataFrame({'category':data_category,'sentence':data_sentence})



X = data.iloc[:, 1].values
y = data.iloc[:, 0].values



# Creating the Bag of Words model
cv = CountVectorizer(max_features = 3000)
X = cv.fit_transform(X).toarray()


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Fitting Naive Bayes to the Training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

correct = 0
wrong = 0

for i,j in zip(y_pred,y_test):
	if i == j:
		correct+=1
	else:
		wrong+=1


accuracy = correct/(correct+wrong)

print('Accuracy: {0:.2f}%'.format(accuracy*100))