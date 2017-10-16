import os
import nltk
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from itertools import *
import copy

trivialTokenizer = RegexpTokenizer(r"\d+|Mr\.|Mrs\.|Dr\.|\b[A-Z]\.|[a-zA-Z_]+-[a-zA-Z_]+-[a-zA-Z_]+|[a-zA-Z_]+-[a-zA-Z_]+|[a-zA-Z_]+|--|'s|'t|'d|'ll|'m|'re|'ve|[.,:!?;\"'()\[\]&@#-]")

WORD_TYPE_COUNT = 4000

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

freqDist = nltk.FreqDist(trainingData)
commonFreqDist = nltk.FreqDist(trainingData).most_common(WORD_TYPE_COUNT)

bigrams = nltk.bigrams(trainingData)

# Create an initialized model where everything is 0
model = {}

# For each bigram, we sort by the first word
vWordTypes = []
sortedBigramDict = {}
for x in commonFreqDist:
    vWordTypes.append(x[0])

for word in bigrams:
    x = False
    y = False
    if word[0] in vWordTypes:
        x = True 
    if word[1] in vWordTypes:
        y = True

    tempList = []

    if x == True and y == True:
        if word[0] in sortedBigramDict:
            tempList = sortedBigramDict[word[0]]
        tempList.append(word[1])
        sortedBigramDict[word[0]] = tempList

# print(sortedBigramDict) 'leisurely': ['walk', 'turns', 'turns', 'blah', 'etc'], 

# Creating an empty slate
vTypeDict = {}
for v in vWordTypes:
    vTypeDict[v] = 0

for v in vWordTypes:
    # Copy temporary dictionary
    tempDict = {}
    tempDict = copy.copy(vTypeDict)

    # Copy sorted bigram list
    if v in sortedBigramDict:
        tempList = []
        tempList = copy.copy(sortedBigramDict[v])
    for x in tempList:
        tempDict[x] += 1

    model[v] = tempDict

'''
for dictionary in model.items():
    if dictionary[0] in sortedBigramDict:
        termList = sortedBigramDict[dictionary[0]]
        #print('Key: ' + dictionary[0] + ' | Value: ' + str(termList))
        for value in termList:
            print(dictionary[0], value)

print(model['plot'])
    '''
        #for term in termList:
            #if dictionary[1] == term:
            #   dictionary[1][term] += 1


'''
for key in model.items():
    if key[0] in sortedBigramDict:
        termList = sortedBigramDict[key[0]]
        #print(x[0]) ('term')
        #print(x[1]) ['asdf', ...]
        for innerKeys in key[1].items():
            for value in termList:
                #print(innerKeys[1])
                innerKeys[0] += 1

print(model['plot'])
'''

'''
# Create a frequency distribution using the tokenized training data
commonFreqDist = nltk.FreqDist(nltk.bigrams(trainingData)).most_common(WORD_TYPE_COUNT)

# Create a blank list and append 'WORD_TYPE_COUNT' most common bigrams
bigramList = []

for bigram in commonFreqDist:
    bigramList.append(bigram[0])

bigrams = list(nltk.bigrams(trainingData))

# Create a dictionary of words and their counts (or probabilities if we manage) of words that follow after
model = defaultdict(lambda: defaultdict(lambda: 0))

for wordOne, wordTwo in bigramList:
    model[wordOne][wordTwo] += 1

for wordOne in model:
    divisibleCount = float(sum(model[wordOne].values()))
    for wordTwo in model:
        model[wordOne][wordTwo] /= divisibleCount

print(model)
'''