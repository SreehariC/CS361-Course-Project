#importing all the needed libraries
import pandas as pd       
import nltk
import sklearn
import sklearn_crfsuite
import scipy.stats
import math, string, re

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from itertools import chain
from sklearn.preprocessing import MultiLabelBinarizer

def word2features(sent, i):
    word = sent[i][1]

    features = {
        'bias': 1.0,
        'word': word,
        'len(word)': len(word),
        'word[:4]': word[:4],
        'word[:3]': word[:3],
        'word[:2]': word[:2],
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[-4:]': word[-4:],
        'word.lower()': word.lower(),
        'word.stemmed': re.sub(r'(.{2,}?)([aeiougyn]+$)',r'\1', word.lower()),
        'word.ispunctuation': (word in string.punctuation),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1][1]
        features.update({
            '-1:word': word1,
            '-1:len(word)': len(word1),
            '-1:word.lower()': word1.lower(),
            '-1:word.stemmed': re.sub(r'(.{2,}?)([aeiougyn]+$)',r'\1', word1.lower()),
            '-1:word[:3]': word1[:3],
            '-1:word[:2]': word1[:2],
            '-1:word[-3:]': word1[-3:],
            '-1:word[-2:]': word1[-2:],
            '-1:word.isdigit()': word1.isdigit(),
            '-1:word.ispunctuation': (word1 in string.punctuation),
        })
    else:
        features['BOS'] = True

    if i > 1:
        word2 = sent[i-2][1]
        features.update({
            '-2:word': word2,
            '-2:len(word)': len(word2),
            '-2:word.lower()': word2.lower(),
            '-2:word[:3]': word2[:3],
            '-2:word[:2]': word2[:2],
            '-2:word[-3:]': word2[-3:],
            '-2:word[-2:]': word2[-2:],
            '-2:word.isdigit()': word2.isdigit(),
            '-2:word.ispunctuation': (word2 in string.punctuation),
            })

    if i < len(sent)-1:
        word1 = sent[i+1][1]
        features.update({
            '+1:word': word1,
            '+1:len(word)': len(word1),
            '+1:word.lower()': word1.lower(),
            '+1:word[:3]': word1[:3],
            '+1:word[:2]': word1[:2],
            '+1:word[-3:]': word1[-3:],
            '+1:word[-2:]': word1[-2:],
            '+1:word.isdigit()': word1.isdigit(),
            '+1:word.ispunctuation': (word1 in string.punctuation),
        })

    else:
        features['EOS'] = True
    if i < len(sent) - 2:
        word2 = sent[i+2][1]
        features.update({
            '+2:word': word2,
            '+2:len(word)': len(word2),
            '+2:word.lower()': word2.lower(),
            '+2:word.stemmed': re.sub(r'(.{2,}?)([aeiougyn]+$)',r'\1', word2.lower()),
            '+2:word[:3]': word2[:3],
            '+2:word[:2]': word2[:2],
            '+2:word[-3:]': word2[-3:],
            '+2:word[-2:]': word2[-2:],
            '+2:word.isdigit()': word2.isdigit(),
            '+2:word.ispunctuation': (word2 in string.punctuation),
        })

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [word[2] for word in sent]

def sent2tokens(sent):
    return [word[0] for word in sent]


# Define a function to parse each line of the dataset
def parse_line(line):
    parts = line.strip().split(',')  # Split the line by comma
    if(parts[0].isdigit() == False):
        return None
    index = int(parts[0])            # Convert index to integer
    word = parts[1]                   # Extract the word
    pos_tag = parts[2]                # Extract the POS tag
    return index, word, pos_tag       # Return the parsed components

# Read the dataset file line by line and parse each line
#importing all the needed libraries
import pandas as pd       
import nltk
import sklearn
import sklearn_crfsuite
import scipy.stats
import math, string, re

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from itertools import chain
from sklearn.preprocessing import MultiLabelBinarizer

def word2features(sent, i):
    word = sent[i][1]

    features = {
        'bias': 1.0,
        'word': word,
        
    }
    if i > 0:
        word1 = sent[i-1][1]
        features.update({
            '-1:word': word1,
            
        })
    else:
        features['BOS'] = True

    # if i > 1:
    #     word2 = sent[i-2][1]
    #     features.update({
    #         '-2:word': word2,
    #         '-2:len(word)': len(word2),
    #         '-2:word.lower()': word2.lower(),
    #         '-2:word[:3]': word2[:3],
    #         '-2:word[:2]': word2[:2],
    #         '-2:word[-3:]': word2[-3:],
    #         '-2:word[-2:]': word2[-2:],
    #         '-2:word.isdigit()': word2.isdigit(),
    #         '-2:word.ispunctuation': (word2 in string.punctuation),
    #         })

    if i < len(sent)-1:
        word1 = sent[i+1][1]
        features.update({
            '+1:word': word1,
    #         '+1:len(word)': len(word1),
    #         '+1:word.lower()': word1.lower(),
    #         '+1:word[:3]': word1[:3],
    #         '+1:word[:2]': word1[:2],
    #         '+1:word[-3:]': word1[-3:],
    #         '+1:word[-2:]': word1[-2:],
    #         '+1:word.isdigit()': word1.isdigit(),
    #         '+1:word.ispunctuation': (word1 in string.punctuation),
         })

    # else:
    #     features['EOS'] = True
    # if i < len(sent) - 2:
    #     word2 = sent[i+2][1]
    #     features.update({
    #         '+2:word': word2,
    #         '+2:len(word)': len(word2),
    #         '+2:word.lower()': word2.lower(),
    #         '+2:word.stemmed': re.sub(r'(.{2,}?)([aeiougyn]+$)',r'\1', word2.lower()),
    #         '+2:word[:3]': word2[:3],
    #         '+2:word[:2]': word2[:2],
    #         '+2:word[-3:]': word2[-3:],
    #         '+2:word[-2:]': word2[-2:],
    #         '+2:word.isdigit()': word2.isdigit(),
    #         '+2:word.ispunctuation': (word2 in string.punctuation),
    #     })

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [word[2] for word in sent]

def sent2tokens(sent):
    return [word[0] for word in sent]


# Define a function to parse each line of the dataset
def parse_line(line):
    parts = line.strip().split(',')  # Split the line by comma
    if(parts[0].isdigit() == False):
        return None
    index = int(parts[0])            # Convert index to integer
    word = parts[1]                   # Extract the word
    pos_tag = parts[2]                # Extract the POS tag
    return index, word, pos_tag       # Return the parsed components






# Print the list of lists (sents)
# for i, sent in enumerate(sents, start=1):
#     print(f"Sentence {i}:")
#     for index, word, pos_tag in sent:
#         print(f"  Index: {index}, Word: {word}, POS Tag: {pos_tag}")

#extracting features from all the sentences
def convert_data(data,num=0):
    converted_data = []
    index = 1
    for sentence in data:
        if(len(sentence)==0): continue
        if(sentence[0]=="</s>"): 
            index=1
            num+=1
            continue
        if sentence[0]=='<s>' or sentence[1].endswith("START") :
            index-=1
            continue
        else:
            converted_data.append(f"{index},{sentence[0]},{sentence[1]}")
        index = 1 if index == len(data) else index + 1
    return converted_data,num

# Function to write converted data to a file
def write_to_file(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(item + '\n')

# Function to read data from a file
def read_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = [line.strip().split() for line in file.readlines()]
    return data
input_filename = 'most_freq_pos.txt'
data = read_from_file(input_filename)
# Convert data
converted_data,num = convert_data(data)
print(f"num of sentences: {num}")
# Write converted data to an output file
output_filename = 'hindi_train1.txt'
write_to_file(converted_data, output_filename)
# Read the dataset file line by line and parse each line


# print('Test set classification report: \n\n{}'.format(metrics.flat_classification_report(ytest, ypred, labels=sorted_labels, digits=3)))
dataset = []
with open('hindi_train1.txt', 'r',encoding='utf8') as file:
    for line in file:
        if line.strip() == ",,":  # Skip empty lines
            continue
        if(parse_line(line) == None):continue
        index, word, pos_tag = parse_line(line)
        dataset.append((index, word, pos_tag))


# Iterate over the dataset
sents=[]
prev_index=None
# Initialize variables to track current sentence
current_sent = []
    
for index, word, pos_tag in dataset:
    # Check if the index is None or different from the previous index
    if prev_index is None or index != prev_index+1:
        # Start a new sentence
        if current_sent:
            # Append the current sentence to sents list
            sents.append(current_sent)
        current_sent = []  # Reset the current sentence
    #else:print("hi")
    # Append the tuple (index, word, POS tag) to the current sentence
    current_sent.append((index, word, pos_tag))
    prev_index = index  # Update the previous index

# Append the last sentence to sents list
if current_sent:
    sents.append(current_sent)

# Print the list of lists (sents)
# for i, sent in enumerate(sents, start=1):
#     print(f"Sentence {i}:")
#     for index, word, pos_tag in sent:
#         print(f"  Index: {index}, Word: {word}, POS Tag: {pos_tag}")

#extracting features from all the sentences
import random
import time
train_sents=[]
test_sents=[]
for i in range(len(sents)):
    if(random.random()>0.1):
        train_sents.append(sents[i])
    else:
        test_sents.append(sents[i])


Xtrain = [sent2features(s) for s in train_sents]
ytrain = [sent2labels(s) for s in train_sents]
print(ytrain[0])

Xtest = [sent2features(s) for s in test_sents]
ytest = [sent2labels(s) for s in test_sents]
print(Xtrain[0])
print("Legnth of Training Data: ",len(train_sents))

start=time.time()                            
crf = sklearn_crfsuite.CRF(
    algorithm = 'lbfgs',
    c1 = 0.25,
    c2 = 0.3,
    max_iterations = 100,
    all_possible_transitions=True
)
crf.fit(Xtrain, ytrain)                  
#training the model
end=time.time()
print("Time taken to train the model: ",end-start)

#obtaining metrics such as accuracy, etc. on the train set
labels = list(crf.classes_)

ypred = crf.predict(Xtrain)
print('F1 score on the train set = {}\n'.format(metrics.flat_f1_score(ytrain, ypred, average='weighted', labels=labels)))
print('Accuracy on the train set = {}\n'.format(metrics.flat_accuracy_score(ytrain, ypred)))

sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
# print('Train set classification report: \n\n{}'.format(metrics.flat_classification_report(
# ytrain, ypred, labels=sorted_labels, digits=3)))
#obtaining metrics such as accuracy, etc. on the test set
ypred = crf.predict(Xtest)

print('F1 score on the test set = {}\n'.format(metrics.flat_f1_score(ytest, ypred,
average='weighted', labels=labels)))
print('Accuracy on the test set = {}\n'.format(metrics.flat_accuracy_score(ytest, ypred)))

sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
# print('Test set classification report: \n\n{}'.format(metrics.flat_classification_report(ytest, ypred, labels=sorted_labels, digits=3)))