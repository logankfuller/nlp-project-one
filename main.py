import os
import nltk
from nltk.tokenize import RegexpTokenizer

trivialTokenizer = RegexpTokenizer(r"\d+|Mr\.|Mrs\.|Dr\.|\b[A-Z]\.|[a-zA-Z_]+-[a-zA-Z_]+-[a-zA-Z_]+|[a-zA-Z_]+-[a-zA-Z_]+|[a-zA-Z_]+|--|'s|'t|'d|'ll|'m|'re|'ve|[.,:!?;\"'()\[\]&@#-]")

WORD_TYPE_COUNT = 10

def file_input(userPath):
   path = input(userPath)
   if os.path.isdir(path):
      return path
   else:
      return file_input('Enter an existing file directory: ')

def probability(data):
   for item in data:
      print(item)
      print(data[item], str('/'), len(trainingData), '=', data[item]/len(trainingData))

directory = file_input('Enter a directory: ')

trainingData = []

for file in os.listdir(directory):
    if file.endswith('.txt'):
        fileData = open(directory + '\\' + file).read()
        trainingData += fileData
    else:
        print('Found non-txt file: ' + file)

trainingData = ''.join(trainingData)

# Begin tokenizing
trainingData = trivialTokenizer.tokenize(trainingData)

# Create a frequency distribution using the tokenized training data
freqDist = nltk.FreqDist(trainingData)
commonFreqDist = nltk.FreqDist(trainingData).most_common(WORD_TYPE_COUNT)

bigrams = list(nltk.bigrams(trainingData))
bigramFreqDist = nltk.FreqDist(bigrams)