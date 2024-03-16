import numpy as np
import pandas as pd
from hmmlearn import hmm
import scipy.stats as st



test = pd.read_csv('../Dataset/TamilDigits-MFCC/Test.csv')
train = pd.read_csv('../Dataset/TamilDigits-MFCC/Train.csv')

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

models = {}

for key in trainLable.keys():
    model = hmm.GMMHMM(verbose=False,n_components=50,n_iter=10000)
    print(trainLable[key].shape)
    feature = np.ndarray(shape=(1, trainLable[key].shape[1]))
    for list_feature in trainLable[key]:
            feature = np.vstack((feature, list_feature))
    obj = model.fit(feature)
    models[key] = obj

correct = 0
total = 0

for index, row in test.iterrows():
    if(index == 0):
        continue
    digit = row['digit']
    row.drop('digit', inplace=True)
    testrow = np.array(row)
    predict = -int(1e9)
    predict_digit = -1
    for key in models.keys():
        score = models[key].score(testrow.reshape(1, -1))
        if score > predict:
            predict = score
            predict_digit = key
    if predict_digit == digit:
        correct += 1
    total += 1

print(correct, total, correct/total)
    