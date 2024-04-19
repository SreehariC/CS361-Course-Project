#!/usr/bin/env python
# -*- coding: utf-8 -*-


from utils.exceptions import FileFormatError


def read_conll_corpus(filename):
    """
    Read a corpus file with a format used in CoNLL.
    """
   

    # Define a function to parse each line of the dataset
    def parse_line(line):
        parts = line.strip().split(',')  # Split the line by comma
        if(parts[0].isdigit() == False):
            return None
        #print(index,word,pos_tag)
        index = int(parts[0])            # Convert index to integer
        word = parts[1]                   # Extract the word
        pos_tag = parts[2]                # Extract the POS tag
        return index, word, pos_tag       # Return the parsed components

    # Read the dataset file line by line and parse each line
    dataset = []
    with open(filename, 'r',encoding='utf8') as file:
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
    return sents