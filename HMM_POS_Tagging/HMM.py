import nltk
import time
import codecs
import os
import sys
from hmmlearn import hmm
import numpy as np
import hmmlearn
import random
from collections import Counter
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Example of stopword removal and lemmatization using NLTK
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer




tags=[]
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
    dataset = []
    with open( os.path.join('..', 'Dataset', 'hindi_train1.txt'), 'r',encoding='utf8') as file:
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

    test=[]
    train=[]
    for i in range(len(sents)):
        if(random.random()<0.1):
            test.append(sents[i])
        else:
            train.append(sents[i])

    print("Number of sentences in training data: ",len(train))
    cnt_unk=0
    dict_tags={}
    wordtypes=[]
    for sent in train:
        for _,word,tag in sent:
            if(tag=="UNK"):
                cnt_unk+=1
            else:
                if tag  not in dict_tags.keys():
                    dict_tags[tag]=0

    print("Number of UNK tags in training data: ",cnt_unk)
    print("Number of tags in training data: ",len(dict_tags.keys()))
    #print(train[0])
    tagscount=[]
    print(dict_tags.keys())
    for i in dict_tags.keys():
        tags.append(i)
        tagscount.append(0)
    for sent in train:
        for _,word,tag in sent:
            if word not in wordtypes: 
                wordtypes.append(word) 
            if tag not in dict_tags.keys():
                dict_tags[tag]=1
                tagscount[tags.index(tag)]+=1
            else:  
                dict_tags[tag]+=1
                tagscount[tags.index(tag)]+=1
    #print(len(wordtypes))
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
        for word in line:
            tag=word[2]
            col_id = wordtypes.index(word[1])
            prev_row_id = row_id
            row_id = tags.index(tag)
            emission_matrix[row_id][col_id] += 1
            if prev_row_id != -1:
                transmission_matrix[prev_row_id][row_id] += 1
        row_id = -1

    for x in range(len(tags)):
        for y in range(len(wordtypes)):
                emission_matrix[x][y] = float(float(emission_matrix[x][y])) / (tagscount[x])

    for x in range(len(tags)):
        for y in range(len(tags)):
                transmission_matrix[x][y] = float(float(transmission_matrix[x][y]))/ (tagscount[x])

    #print(emission_matrix)
    print(time.time() - start_time, "seconds for training using MLE")
    #print(transmission_matrix)
    
    start_time = time.time()

    num_correct=0
    total=0
    prob_small_values = np.linspace(0.000001,1, 30)  # Adjusted range to [0.0001, 0.1]

    # Initialize lists to store F1 scores and accuracies
    f1_scores = []
    accuracies = []

    # Iterate over different values of prob_small
    for prob_small in prob_small_values:
        pred_labels=[]
        true_labels=[]
        
        for j in range(len(test)):
            test_words = []
            pos_tags = []
            line=test[j]
            for word in line:
                test_words.append(word[1])
                pos_tags.append(-1)
            
            viterbi_matrix = []
            viterbi_path = []
            

            for x in range(len(tags)):
                viterbi_matrix.append([])
                viterbi_path.append([])
                for y in range(len(test_words)):
                    viterbi_matrix[x].append(0)
                    viterbi_path[x].append(0)

            # Update viterbi matrix column wise
            for x in range(len(test_words)):
                #print(test_words[x])
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
                true_labels.append(line[x][2])
                pred_labels.append(tags[pos_tags[x]])
                if(line[x][2]==tags[pos_tags[x]]):
                    num_correct+=1
                total+=1
        print("Accuracy: ",num_correct/total)
        # Compute classification report
        report = classification_report(true_labels, pred_labels, output_dict=True,zero_division=0)

        # Extract class labels and F1 scores
        class_labels = list(report.keys())[:-3]
        F1_scores = [report[label]['f1-score'] for label in class_labels]
        mean_f1_score = np.mean(F1_scores)
        print("Mean F1 score: ",mean_f1_score)
        accuracies.append(num_correct/total)
        f1_scores.append(mean_f1_score)
    # Plotting the F1 scores
   # Plotting the F1 scores
    plt.figure(figsize=(10, 6))

# Plot F1 scores
    plt.plot(prob_small_values, f1_scores, label='F1 Score', marker='o', color='blue')

    # Plot accuracies
    plt.plot(prob_small_values, accuracies, label='Accuracy', marker='x', color='red')

    plt.xlabel('prob_small')
    plt.ylabel('Score')
    plt.title('F1 Score and Accuracy vs. prob_small')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Calculate accuracy
    # correct_predictions = sum(1 for true_tags, pred_tags in zip(hmm_output_test, predicted_pos_tags_str) if true_tags == pred_tags)
    # total_predictions = len(hmm_output_test)
    # accuracy = correct_predictions / total_predictions
    # print("Accuracy:", accuracy)

def train_em(prob_small,num_of_states=10):
    start_time = time.time()
    train_data=nltk.corpus.treebank.tagged_sents() # reading the Treebank tagged sentences
    file_output = codecs.open("./output/"+ "self_out", 'a', 'utf-8')

    test=[]
    train=[]
    for i in range(len(train_data)):
        if(random.random()<0.1):
            test.append(train_data[i])
        else:
            train.append(train_data[i])

    print("Number of sentences in training data: ",len(train))
    wordtypes=[]

    print(train_data[0])
    tagscount=[]
    for i in tags:
        tagscount.append(0)
    for sent in train_data:
        for word,tag in sent:
            if word not in wordtypes: 
                wordtypes.append(word) 
            
    
    emission_matrix = []
    transmission_matrix = []
    for x in range(num_of_states):
        emission_matrix.append([])
        sum=0
        for y in range(len(wordtypes)):
            y=random.random()
            emission_matrix[x].append(y)  ## Too many states, can we reduce it?
            sum+=y
        for y in range(len(wordtypes)):
            emission_matrix[x][y]/=sum

    # Update emission and transmission matrix with appropriate counts
    PI=[]
    sum=0
    for i in range(num_of_states):
        y=random.random()
        sum+=y
        PI.append(y)
    for i in range(num_of_states): 
        PI[i]/=sum

    for x in range(num_of_states):
        transmission_matrix.append([])
        sum=0
        for y in range(num_of_states):
            y=random.random()
            transmission_matrix[x].append(y)
            sum+=y
        for y in range(num_of_states):
            
            transmission_matrix[x][y]/=sum

    co=0
    denom_transimission=[]
    for i in range(num_of_states):
        denom_transimission.append(0)  
    cnt=0
    for line in train:
        cnt+=1
        if(cnt==5): break
        alpha=[]
        start=0
        co=0
        C=[]
        for word,tag in line:
            temp=[]
            co=0
            if(start==0):
                for i in range(num_of_states):
                    temp.append(PI[i]*emission_matrix[i][wordtypes.index(word)])
                    co+=temp[i]
                C.append(co)
                for i in range(num_of_states):
                    temp[i]/=co
                alpha.append(temp)
                start=1
            else:
                for i in range(num_of_states):
                    temp.append(0)
                    for j in range(num_of_states):
                        temp[i]+=alpha[-1][j]*transmission_matrix[j][i]
                    temp[i]*=emission_matrix[i][wordtypes.index(word)]
                    co+=temp[i]
                C.append(co)
                for i in range(num_of_states):
                    temp[i]/=co
                alpha.append(temp)
        beta=[]
        start=0
        co=0
        temp=[]
        for _,_ in line:
            beta.append([])
        for i in range(num_of_states):
            temp.append(1)
        C.pop()
        beta[-1]=temp
        idx=len(C)-1
        #print(C[-1])
        
        for word,tag in reversed(line):
            co=0
            if(word==line[-1][0]):continue
            temp=[]
            for i in range(num_of_states):
                temp.append(0)
                for j in range(num_of_states):
                    temp[i]+=transmission_matrix[i][j]*emission_matrix[j][wordtypes.index(word)]*beta[idx+1][j]
                co+=temp[i]
            for i in range(num_of_states):
                temp[i]/=co
            if(len(C)>0):
             C.pop()
            beta[idx]=temp
            idx-=1
        file_output.write(str(alpha))
        file_output.write('******************\n')
        file_output.write(str(beta))
        gamma=[]
        gammat=[]      
        for i in range(len(line)-1):
            rows=num_of_states
            cols=num_of_states
            matrix = [[0 for j in range(cols)] for i in range(rows)]
            temp=[]
            for j in range(num_of_states):
                gj=0
                for k in range(num_of_states):
                    matrix[j][k]=alpha[i][j]*transmission_matrix[j][k]*emission_matrix[k][wordtypes.index(line[i+1][0])]*beta[i+1][k]
                    gj+=matrix[j][k]
                temp.append(gj)
            gamma.append(matrix)
            gammat.append(temp)

        temp1=[]      
        for i in range(num_of_states):
            temp1.append(alpha[-1][i])
        gammat.append(temp1)

        for i in range(num_of_states):
            PI[i]+=gammat[0][i]
        for i in range(num_of_states):
            denom=0
            for j in range(len(line)-1):
                denom+=gammat[j][i]
            for j in range(num_of_states):
                numer=0
                for k in range(len(line)-1):
                    numer+=gamma[k][i][j]
                transmission_matrix[i][j]=+numer
            denom_transimission[i]+=denom
        
            for i in range(num_of_states):
                denom=0
                for j in range(len(line)):
                    denom+=gammat[j][i]
                for j in range(len(wordtypes)):
                    numer=0
                    for k in range(len(line)):
                        if(line[k][0]==wordtypes[j]):
                            numer+=gammat[k][i]
                    emission_matrix[i][j]+=numer

    for i in range(num_of_states):  
        PI[i]/=len(train)
        for j in range(num_of_states):
            transmission_matrix[i][j]/=denom_transimission[i]
        for j in range(len(wordtypes)):
            emission_matrix[i][j]/=denom_transimission[i]
    
    print(time.time() - start_time, "seconds for training using MLE")
    print(transmission_matrix)

    
    start_time = time.time()

    num_correct=0
    total=0
    file_output = codecs.open("./output/"+ "Unsupervided_tags.txt", 'a', 'utf-8')
    cnt=0
    for j in range(len(test)):
        cnt+=1
        if(cnt==5): break
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
        for i, x in enumerate(pos_tags):
            file_output.write(test_words[i] + "_" + tags[x] + " ")
        file_output.write(" ._.\n")	
    print("Accuracy: ",num_correct/total)




def data_visualize():
    nltk.download('punkt')
    nltk.download('indian')

    from nltk.corpus import indian

    # Load the Hindi POS tagged sentences
    hindi_tagged_sents = indian.tagged_sents('hindi.pos')
    print(len(hindi_tagged_sents))
    # Flatten the list of tagged words
    tagged_words = [word for sent in hindi_tagged_sents for word in sent]

    # Extract just the POS tags
    pos_tags = [tag for (word, tag) in tagged_words]

    # Count the occurrences of each POS tag
    tag_counts = Counter(pos_tags)

    # Plot the distribution of POS tags
    plt.figure(figsize=(10, 6))
    plt.bar(tag_counts.keys(), tag_counts.values())
    plt.title('Distribution of POS tags in Hindi dataset')
    plt.xlabel('POS Tags')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()

# Example of custom tokenization using regular expressions
import re

def custom_tokenize(text):
    # Tokenize based on non-alphanumeric characters and remove any empty strings
    tokens = re.split(r'\W+', text)
    tokens = [token for token in tokens if token]
    return tokens


def preprocess_text(text):
    # Tokenize the text
    
    tokens = custom_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('hindi'))
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

import nltk
from nltk.tag import hmm

def library_imp():
    # Load the Hindi tagged sentences
    hindi_tagged_sents = nltk.corpus.indian.tagged_sents('hindi.pos')

    # Split the dataset into train and test sets
    train_size = int(0.9*len(hindi_tagged_sents))
    train_sents = []
    test_sents = []
    for i in hindi_tagged_sents:
        if(random.random()>0.9):
            test_sents.append(i)
        else:
            train_sents.append(i)

    # Train the HMM-based POS tagger
    tagger = hmm.HiddenMarkovModelTrainer().train(train_sents)

    # Evaluate the tagger on the test set
    accuracy = tagger.evaluate(test_sents)
    print("Accuracy:", accuracy)



if __name__ == "__main__":
    # Function to convert data into the desired format
    def convert_data(data):
        converted_data = []
        index = 1
        for sentence in data:
            if(len(sentence)==0): continue
            if(sentence[0]=="</s>"): 
                index=1
                continue
            if sentence[0]=='<s>' or sentence[1].endswith("START") :
                index-=1
                continue
            else:
                converted_data.append(f"{index},{sentence[0]},{sentence[1]}")
            index = 1 if index == len(data) else index + 1
        return converted_data

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
    
    input_filename = os.path.join('..', 'Dataset', 'most_freq_pos.txt')
    data = read_from_file(input_filename)
    # Convert data
    converted_data = convert_data(data)

    # Write converted data to an output file
    output_filename = os.path.join('..', 'Dataset', 'hindi_train1.txt')
    write_to_file(converted_data, output_filename)
    train()
    

    


            


    