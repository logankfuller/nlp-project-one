import os
import nltk
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from itertools import *
import copy
import random
import pickle
import pprint
from collections import OrderedDict

trivialTokenizer = RegexpTokenizer(r"\d+|Mr\.|Mrs\.|Dr\.|\b[A-Z]\.|[a-zA-Z_]+-[a-zA-Z_]+-[a-zA-Z_]+|[a-zA-Z_]+-[a-zA-Z_]+|[a-zA-Z_]+|--|'s|'t|'d|'ll|'m|'re|'ve|[.,:!?;\"'()\[\]&@#-]")

WORD_TYPE_COUNT = 12000

# Create an initialized model where everything is 0
model = {}

def dictCopy(item):
    temp = {}
    newCopy = {}
    temp = item
    newCopy = temp
    return newCopy

def removeKey(mDict, key):
    r = dict(mDict)
    del r[key]
    return r

def weighted_choice(choices, prob):
    r = random.uniform(0.0, prob)
    upto = 0
    for c in choices.items():
        if upto + c[1] >= r:
            return c[0]
        upto += c[1]

def file_input(userPath):
   path = input(userPath)
   if os.path.isdir(path):
      return path
   else:
      return file_input('Enter an existing file directory: ')

# SANITY CHECK (THERE IS NO SANITY LEFT IN US)
def getMaxProb(unigram):
    probability = 0
    for word in model[unigram].items():
        probability += word[1]
    return probability

userEntry = 'N'
if os.path.exists('model.pickle'):
    userEntry = input("Use pickled dictionary? (Y/N)")

if userEntry == 'Y' or userEntry == 'y':
    with open('model.pickle', 'rb') as handle:
        model = pickle.load(handle)
else:
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
        tempDict = dictCopy(vTypeDict)

        # Copy sorted bigram list
        if v in sortedBigramDict:
            tempList = []
            tempList = dictCopy(sortedBigramDict[v])
        for x in tempList:
            if x != v:
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

    with open('model.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

#pprint.pprint(model['these'])
    #print(model['plot'])

# Sentence generation
print("Generating sentence...")

# Pick a random word
sentence = []

# Grab random first word and append
currentWord = random.choice(list(model.keys()))
sentence.append(currentWord)

print('Initial Rand: ', currentWord)

sentence_finished = False
while not sentence_finished:
    r = random.random()
    accumulator = .0

    #orderedModel = OrderedDict(sorted(model[currentWord].items(), key=lambda kv: kv[1], reverse=True))

    for word in model[currentWord]:
        accumulator += model[currentWord][word]

        if accumulator >= r:
            if word != currentWord:
                sentence.append(word)
                currentWord = word
                break

    if(len(sentence) >= 30):
        sentence_finished = True

print(' '.join(sentence))

print(sentence)
sentenceBigrams = nltk.bigrams(sentence)

sentenceProbability = 1.0
for bigram in sentenceBigrams:
    print(model[bigram[0]][bigram[1]])
    sentenceProbability *= model[bigram[0]][bigram[1]]

print(sentenceProbability)








# Calculating probability of sentences

sentenceOne = "Go to the park and meet girls"
sentenceTwo = "No, I'm not paying for the damage to your car"