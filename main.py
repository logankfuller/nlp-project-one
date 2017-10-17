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

unigramCount = len(trainingData)

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

    # ------------ Smoothing
    
    # Get list of bigrams with counts 1-9
    lowFreqBigramCount = 0
    for bigram in tempDict.items():
        if bigram[1] <= 9 and bigram[1] > 0:
            lowFreqBigramCount += 1

    # Increase each bigram count by 1/lowFreqBigramCount
    additiveBigram = 0
    if lowFreqBigramCount != 0:
        additiveBigram = 1/lowFreqBigramCount
        for bigram in tempDict.items():
            tempDict[bigram[0]] += additiveBigram

    # Increase each unigram count by V/lowFreqBigramCount
    currentUnigramCount = 0
    additiveUnigram = 0
    if lowFreqBigramCount != 0:
        additiveUnigram = WORD_TYPE_COUNT / lowFreqBigramCount
        currentUnigramCount = freqDist[v] + additiveUnigram
        
    # Probabilities
    for bigram in tempDict.items():
        if currentUnigramCount != 0:
            tempDict[bigram[0]] /= currentUnigramCount

    model[v] = tempDict

#print(model['plot'])

# SANITY CHECK (THERE IS NO SANITY LEFT IN US)
probability = 0
for word in model['plot'].items():
    probability += word[1]
print(probability)