import os
import nltk
from nltk.tokenize import RegexpTokenizer
import random
import pickle
import pprint

def file_input(userPath):
    path = input(userPath)
    if os.path.isdir(path):
        return path
    else:
        return file_input('Enter an existing file directory: ')

userEntry = 'N'
if os.path.exists('model.pickle'):
    userEntry = input("Use pickled dictionary? (Y/N)")

if userEntry == 'Y' or userEntry == 'y':
    with open('model.pickle', 'rb') as handle:
        model = pickle.load(handle)
else:
    directory = file_input('Enter a directory: ')

    trivialTokenizer = RegexpTokenizer(r"\d+|Mr\.|Mrs\.|Dr\.|\b[A-Z]\.|[a-zA-Z_]+-[a-zA-Z_]+-[a-zA-Z_]+|[a-zA-Z_]+-[a-zA-Z_]+|[a-zA-Z_]+|--|'s|'t|'d|'ll|'m|'re|'ve|[.,:!?;\"'()\[\]&@#-]")

    trainingData = []

    for file in os.listdir(directory):
        if file.endswith('.txt'):
            fileData = open(directory + '\\' + file).read()
            trainingData += fileData
        else:
            print('Found non-txt file: ' + file)

    trainingData = ''.join(trainingData)

    # Begin tokenizing
    trainingTokens = trivialTokenizer.tokenize(trainingData)

    WORD_TYPE_COUNT = 4000

    #unigrams = nltk.FreqDist(trainingTokens)
    commonFreqDist = nltk.FreqDist(trainingTokens).most_common(WORD_TYPE_COUNT)
    bigramFreqDist = nltk.FreqDist(nltk.bigrams(trainingTokens))

     # Create an initialized model where everything is 0
    model = {}

    # Add each frequent word into a model with a None value
    # freqWord[0] = word, [1] = count
    for freqWord in commonFreqDist:
        rowToAdd = {}
        # Add a copy of the model to each word type already in the model (this creates an identical 2d array)
        for tempWord in commonFreqDist:
            rowToAdd[tempWord[0]] = 0
        model[freqWord[0]] = rowToAdd
    
    # Clean up the bigram freq dist 
    cleanedBigramFreqDist = []
    for bigram in bigramFreqDist.items():
        if(bigram[0][0] in model and bigram[0][1] in model):
            cleanedBigramFreqDist.append(bigram)

    unigramList = []
    for item in cleanedBigramFreqDist:
        unigramList.append(item[0][0])
        unigramList.append(item[0][1])
    
    unigrams = nltk.FreqDist(unigramList)
    #print(unigrams.items())

    # How to access cleanedBigramFreqDist
    # print(cleanedBigramFreqDist[0][0][0])

    # Now to populate the inner values with actual bigram data
    # bigram[0] returns first word, bigram[1] returns second word, value returns bigram count
    for bigram, value in cleanedBigramFreqDist:
        model[bigram[0]][bigram[1]] = value

    # ------------ Smoothing ------------- #
    
    # {'test': {'test1': 0}}
    # outer = 'test', inner = 'test1'
    
    # Get list of bigrams with counts 1-9
    lowFreqBigramCount = 0
    for outer in model:
        for inner in model[outer]:
            value = 0
            value = model[outer][inner]
            if value < 10 and value > 0:
                lowFreqBigramCount += 1
    
    # Increase each bigram count by 1/lowFreqBigramCount
    additiveBigram = 0
    additiveBigram = 1/lowFreqBigramCount
    for outer in model:
        for inner in model[outer]:
            model[outer][inner] += additiveBigram

    print(model['the'])
    
    # Increase each unigram count by WORD_TYPE_COUNT/lowFreqBigramCount
    additiveUnigram = 0
    additiveUnigram = WORD_TYPE_COUNT / lowFreqBigramCount
    for unigram in unigrams:
        unigrams[unigram] += additiveUnigram

    #print(sum(model['asks'].values()))
    
    # Calculate the probabilities of each bigram count
    for outer in model:
        for inner in model[outer]:
            #print(str(model[outer][inner]) + ' : ' + str(unigrams[outer]))
            model[outer][inner] /= unigrams[outer]
    
    '''
    #with open('model.pickle', 'wb') as handle:
    #    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    '''