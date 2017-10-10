import os, nltk
from nltk.tokenize import RegexpTokenizer

trivialTokenizer = RegexpTokenizer(r"\d+|Mr\.|Mrs\.|Dr\.|\b[A-Z]\.|[a-zA-Z_]+-[a-zA-Z_]+-[a-zA-Z_]+|[a-zA-Z_]+-[a-zA-Z_]+|[a-zA-Z_]+|--|'s|'t|'d|'ll|'m|'re|'ve|[.,:!?;\"'()\[\]&@#-]")

def file_input(userPath):
   path = input(userPath)
   if os.path.isdir(path):
      return path
   else:
      return file_input('Enter an existing file directory: ')

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
print(len(trainingData))

# Create a frequency distribution using the tokenized training data
freqDist = nltk.FreqDist(trainingData)
print(list(freqDist.keys())[:10])
print(freqDist["the"] / len(freqDist.keys()))

'''
freq_training = nltk.FreqDist(trainingData)
cfreq_training = nltk.ConditionalFreqDist(nltk.bigrams(trainingData))
cprob_training = nltk.ConditionalProbDist(cfreq_training, nltk.MLEProbDist)

print(freq_training.most_common(10), end='\n\n')
print(cfreq_training["the"].most_common(10), end='\n\n')
print(cprob_training["my"].samples(), end='\n\n')

word = "Doctor"
for index in range(10):
   word = cprob_training[word].generate()
   print(word, end=" ")

print(list(freq_training.keys())[:20])
print(freq_training.most_common(20))
'''
