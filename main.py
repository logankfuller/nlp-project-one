import os, re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
import random
import pickle
import pprint

def file_input(userPath):
	path = input(userPath)
	if os.path.isdir(path):
		return path
	else:
		return file_input('Enter an existing file directory: ')

def getMaxProb(unigram):
	probability = 0
	for word in model[unigram].items():
		probability += word[1]
	return probability
	
def sentenceProbability(s):
	sBigrams = nltk.bigrams(s)
	sProb = 1.0
	
	for bigram in sBigrams:
		sProb *= model[bigram[0]][bigram[1]]
	print('> Sentence Probability: ' + str(sProb))
	
def compareSentences(s1, s2):
	s1Bigrams = nltk.bigrams(trivialTokenizer.tokenize(s1))
	s2Bigrams = nltk.bigrams(trivialTokenizer.tokenize(s2))
	s1prob = 1.0
	s2prob = 1.0
	
	for bigram in s1Bigrams:
		s1prob *= model[bigram[0]][bigram[1]]
	
	for bigram in s2Bigrams:
		s2prob *= model[bigram[0]][bigram[1]]
		
	if s1prob > s2prob:
		print('> S1 Probability: ' + str(s1prob) + '| S2 Probability: ' + str(s2prob))
		print('> Sentence 1 has higher probability')
	else:
		print('> S1 Probability: ' + str(s1prob) + '| S2 Probability: ' + str(s2prob))
		print('> Sentence 2 has higher probability')

def generateSentence():
	# Sentence generation
	print('> ----- Sentence Generation -----')
	cont = 'Y'
	while cont == 'Y' or cont == 'y':
		size = int(input('> Input Sentence Size [Integer]: '))
	
		print("> Generating sentence...")

		# Pick a random word
		sentence = []
		qSentence = ''

		# Grab random first word and append
		currentWord = random.choice(list(model.keys()))
		sentence.append(currentWord)

		sentence_finished = False
		while not sentence_finished:
			r = random.uniform(0, 1.0)
			accumulator = .0

			for word in model[currentWord]:
				accumulator += model[currentWord][word]

				if accumulator >= r:
					if word != currentWord:
						sentence.append(word)
						currentWord = word
						break

			if(len(sentence) >= size):
				sentence_finished = True
		
		gSentence = ' '.join(sentence)
		
		print('> Generated sentence: ' + ' '.join(sentence))
		cont = input('> Generate Again [Y/N]: ')
	
def doCompare():
	print('> ----- Sentence Comparison -----')
	cont = 'Y'
	while cont == 'Y' or cont == 'y':
		sent1 = input('> Input Sentence 1: ')
		sent2 = input('> Input Sentence 2: ')
		compareSentences(sent1, sent2)
		cont = input('> Compare Again [Y/N]: ')

 # Create an initialized model where everything is 0
model = {}
trivialTokenizer = RegexpTokenizer(r"\d+|Mr\.|Mrs\.|Dr\.|\b[A-Z]\.|[a-zA-Z_]+-[a-zA-Z_]+-[a-zA-Z_]+|[a-zA-Z_]+-[a-zA-Z_]+|[a-zA-Z_]+|--|'s|'t|'d|'ll|'m|'re|'ve|[.,:!?;\"'()\[\]&@#-]")
	
userEntry = 'N'
if os.path.exists('model.pickle'):
	userEntry = input("> Use pickled dictionary? (Y/N)")

if userEntry == 'Y' or userEntry == 'y':
	with open('model.pickle', 'rb') as handle:
		model = pickle.load(handle)
else:
	directory = file_input('> Enter a directory: ')

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
	#trainingTokens = nltk.word_tokenize(trainingData)

	WORD_TYPE_COUNT = 8000

	unigrams = nltk.FreqDist(trainingTokens)
	bigrams = nltk.bigrams(trainingTokens)
	commonFreqDist = nltk.FreqDist(trainingTokens).most_common(WORD_TYPE_COUNT)
	bigramFreqDist = nltk.FreqDist(nltk.bigrams(trainingTokens))
	
	vWordTypes = []
	for x in commonFreqDist:
		vWordTypes.append(x[0])

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
	unigramCorrector = {}
	for bigram in bigramFreqDist.items():
		x = False
		y = False
		if bigram[0][0] in vWordTypes:
			x = True 
		if bigram[0][1] in vWordTypes:
			y = True

		if x == True and y == True:
			cleanedBigramFreqDist.append(bigram)
		elif x == True and y == False:
			if bigram[0][0] in unigramCorrector:
				unigramCorrector[bigram[0][0]] += 1
			else:
				unigramCorrector[bigram[0][0]] = 1
		
	#print(cleanedBigramFreqDist)
	#print(str(unigrams['M.']) + ':' + str(unigramCorrector['M.']))

	#print(unigrams.items())

	# How to access cleanedBigramFreqDist
	# print(cleanedBigramFreqDist[0][0][0])

	# Now to populate the inner values with actual bigram data
	# bigram[0] returns first word, bigram[1] returns second word, value returns bigram count
	
	for bigram, value in cleanedBigramFreqDist:
		model[bigram[0]][bigram[1]] = value
		
	sums = {}
	for outer in model:
		s = 0
		for inner in model[outer]:
			s += model[outer][inner]
		sums[outer] = s
	#print(str(sums['M.']))
	for s in sums:
		sums[s] /= unigrams[s]
	
	#print(str(sums['M.']))
	#print(sums)

	# ------------ Smoothing ------------- #
	# Smooth bigram (a,b) + i
	# Smooth Unigram count (a) + (V * i)
	# Divide those two for probabilities
	# probability r* = (cnt(a,b) + i) / ((a) + (V * i))
	
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
	
	# Increase each unigram count by WORD_TYPE_COUNT/lowFreqBigramCount
	additiveUnigram = 0
	additiveUnigram = WORD_TYPE_COUNT * additiveBigram
	for unigram in unigrams:
		unigrams[unigram] += additiveUnigram
		if unigram in unigramCorrector:
			unigrams[unigram] -= unigramCorrector[unigram]

	#print(sum(model['asks'].values()))
	
	# Calculate the probabilities of each bigram count
	for outer in model:
		for inner in model[outer]:
			#print(str(model[outer][inner]) + ' : ' + str(unigrams[outer]))
			model[outer][inner] /= unigrams[outer]
	
	with open('model.pickle', 'wb') as handle:
		pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

generateSentence()
doCompare()

