import numpy as np
import pandas as pd
from hmmlearn import hmm
import scipy.stats as st
from scipy.io import wavfile
from python_speech_features import mfcc
import os

def normalize_2d(matrix):
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm  # normalized matrix
    return matrix


test = pd.read_csv('../Dataset/HindiDigitsMFCC/Test.csv')
train = pd.read_csv('../Dataset/HindiDigitsMFCC/Train.csv')

def build_dataset(sound_path='../Dataset/HindiDigits/'):
    files = sorted(os.listdir(sound_path))
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    data = dict()
    n = len(files)
    for i in range(n):
        feature = feature_extractor(sound_path=sound_path + files[i])
        digit = files[i][0]
        if digit not in data.keys():
            data[digit] = []
            x_test.append(feature)
            y_test.append(digit)
        else:
            if np.random.rand() < 0.1:
                x_test.append(feature)
                y_test.append(digit)
            else:
                x_train.append(feature)
                y_train.append(digit)
            data[digit].append(feature)
    return x_train, y_train, x_test, y_test, data

def feature_extractor(sound_path):
    sampling_freq, audio = wavfile.read(sound_path)
    mfcc_features = mfcc(audio, sampling_freq,nfft = 2048,numcep=13,nfilt=13)
    return mfcc_features

test.drop('File Name', axis=1, inplace=True)
train.drop('File Name', axis=1, inplace=True)

testLable ={}
trainLable = {}

for index, row in train.iterrows():
    if(index == 0):
        continue
    if row['digit'] not in trainLable.keys():
        trainLable[row['digit']] = []
    digit = row['digit']
    row.drop('digit', inplace=True)
    trainLable[digit].append(row)

for key in trainLable.keys():
    trainLable[key] = np.array(trainLable[key])

# for index, row in test.iterrows():
#     if(index == 0):
#         continue
#     if row['digit'] not in testLable.keys():
#         testLable[row['digit']] = []
#     digit = row['digit']
#     row.drop('digit', inplace=True)
#     testLable[digit].append(row)

# for key in testLable.keys():
#     testLable[key] = np.array(testLable[key])
x_train, y_train, x_test, y_test, data = build_dataset()

models = {}

print(data.keys())

for key in data.keys():
    model = hmm.GMMHMM(verbose=False,n_components=5,n_iter=10000, covariance_type='diag')
    feature = np.ndarray(shape=(1, 13))
    for list_feature in data[key]:
            feature = np.vstack((feature, list_feature))
    print(feature.shape)
    obj = model.fit(feature)
    models[key] = obj

correct = 0
total = 0

# for index, row in test.iterrows():
#     if(index == 0):
#         continue
#     digit = row['digit']
#     row.drop('digit', inplace=True)
#     testrow = np.array(row)
#     predict = -int(1e9)
#     predict_digit = -1
#     for key in models.keys():
#         score = models[key].score(testrow.reshape(1, -1))
#         if score > predict:
#             predict = score
#             predict_digit = key
    
#     print(predict_digit, digit)
#     if predict_digit == digit:
#         correct += 1
#     total += 1

for i in range(len(x_train)):
    testrow = x_train[i]
    digit = y_train[i]
    predict = -int(1e9)
    predict_digit = -1
    for key in models.keys():
        score = models[key].score(testrow)
        if score > predict:
            predict = score
            predict_digit = key
    if predict_digit == digit:
        correct += 1
    total += 1

print(correct, total, correct/total)
    