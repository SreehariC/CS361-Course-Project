import nltk
import time
# import codecs
import os
import sys
from hmmlearn import hmm
import numpy as np
import hmmlearn
import random
tags=[]

model = hmm.GMMHMM(verbose=False,n_components=100,n_iter=10000)

def max_connect(x, y, viterbi_matrix, emission, transmission_matrix):
	max = -99999
	path = -1
	
	for k in range(len(tags)):
		val = viterbi_matrix[k][x-1] * transmission_matrix[k][y]
		if val * emission > max:
			max = val
			path = k
	return max, path

def train():
    start_time = time.time()
    train_data= nltk.corpus.indian.tagged_sents('hindi.pos')
    # file_output = codecs.open("./output/"+ "self_out", 'a', 'utf-8')

    test=[]
    train=[]
    for i in range(len(train_data)):
        if(random.random()<0.1):
            test.append(train_data[i])
        else:
            train.append(train_data[i])

    print("Number of sentences in training data: ",len(train))
    cnt_unk=0
    dict_tags={}
    wordtypes=[]
    for sent in train:
        for word,tag in sent:
            if(tag=="UNK"):
                cnt_unk+=1
            else:
                if tag  not in dict_tags.keys():
                    dict_tags[tag]=0

    print("Number of UNK tags in training data: ",cnt_unk)
    print("Number of tags in training data: ",len(dict_tags.keys()))
    print(train_data[0])
    tagscount=[]
    print(dict_tags.keys())
    for i in dict_tags.keys():
        tags.append(i)
        tagscount.append(0)
    for sent in train:
        for word,tag in sent:
            if word not in wordtypes: 
                wordtypes.append(word) 
            if tag not in dict_tags.keys():
                dict_tags[tag]=1
                tagscount[tags.index(tag)]+=1
            else:  
                dict_tags[tag]+=1
                tagscount[tags.index(tag)]+=1
    
    emission_matrix = []
    transmission_matrix = []


		
	# Initialize emission matrix
    for x in range(len(tags)):
        emission_matrix.append([])
        for y in range(len(wordtypes)):
            emission_matrix[x].append(0)

    for x in range(len(tags)):
        transmission_matrix.append([])
        for y in range(len(tags)):
            transmission_matrix[x].append(0)
    


    # Update emission and transmission matrix with appropriate counts
    row_id = -1
    for x in range(len(train)):
        line = train[x]
        for word,tag in line:
            col_id = wordtypes.index(word)
            prev_row_id = row_id
            row_id = tags.index(tag)
            emission_matrix[row_id][col_id] += 1
            if prev_row_id != -1:
                transmission_matrix[prev_row_id][row_id] += 1
        row_id = -1

    for x in range(len(tags)):
        for y in range(len(wordtypes)):
            if tagscount[x] != 0:
                emission_matrix[x][y] = float(emission_matrix[x][y]) / tagscount[x]

    for x in range(len(tags)):
        for y in range(len(tags)):
            if tagscount[x] != 0:
                transmission_matrix[x][y] = float(transmission_matrix[x][y]) / tagscount[x]

    print(time.time() - start_time, "seconds for training using MLE")

    
    start_time = time.time()
    num_correct=0
    total=0

    for j in range(len(test)):
        test_words = []
        pos_tags = []
        line=test[j]
        for word,_ in line:
            test_words.append(word)
            pos_tags.append(-1)
        
        viterbi_matrix = []
        viterbi_path = []
        
        prob_small=0.0001

        for x in range(len(tags)):
            viterbi_matrix.append([])
            viterbi_path.append([])
            for y in range(len(test_words)):
                viterbi_matrix[x].append(0)
                viterbi_path[x].append(0)

		# Update viterbi matrix column wise
        for x in range(len(test_words)):
            for y in range(len(tags)):
                if test_words[x] in wordtypes:
                    word_index = wordtypes.index(test_words[x])
                    tag_index = tags.index(tags[y])
                    emission = emission_matrix[tag_index][word_index]
                else:
                    emission = prob_small

                if x > 0:
                    max, viterbi_path[y][x] = max_connect(x, y, viterbi_matrix, emission, transmission_matrix)
                else:
                    max = 1
                viterbi_matrix[y][x] = emission * max

                # Identify the max probability in last column i.e. best tag for last word in test sentence
        maxval = -999999
        maxs = -1
        for x in range(len(tags)):
            if viterbi_matrix[x][len(test_words)-1] > maxval:
                maxval = viterbi_matrix[x][len(test_words)-1]
                maxs = x

        # Backtrack and identify best tags for each words
        for x in range(len(test_words)-1, -1, -1):
            pos_tags[x] = maxs
            maxs = viterbi_path[maxs][x]
        
        for x in range(len(line)):
            if(line[x][1]==tags[pos_tags[x]]):
                num_correct+=1
            total+=1

    print("Accuracy: ",num_correct/total)
    # Calculate accuracy
    # correct_predictions = sum(1 for true_tags, pred_tags in zip(hmm_output_test, predicted_pos_tags_str) if true_tags == pred_tags)
    # total_predictions = len(hmm_output_test)
    # accuracy = correct_predictions / total_predictions
    # print("Accuracy:", accuracy)

if __name__ == "__main__":
    train()


            


    