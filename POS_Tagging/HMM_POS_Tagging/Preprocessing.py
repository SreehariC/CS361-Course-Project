import random

# def create():
#     # Read the input file
#     with open("hindi_training.txt", "r", encoding="utf-8") as f:
#         lines = f.readlines()

#     # Define the probability threshold
#     probability_threshold = 0.05

#     # Function to randomly assign "UNK" tag to words below threshold
#     def assign_unk_tag(word):
#         return "UNK" if random.random() < probability_threshold and word != 'START' and word != 'END' else word

#     # Process each line
#     output_lines = []
#     for line in lines:
#         # Split the line into word and tag
#         word, tag = line.strip().split()
#         # Assign "UNK" tag to some words randomly, excluding start and end tokens
#         tag = assign_unk_tag(tag)
#         # Append the processed line to the output list
#         output_lines.append(f"{word} {tag}")

#     # Write the output to a new file
#     with open("hindi_pos.txt", "w", encoding="utf-8") as f:
#         f.write("\n".join(output_lines))

#     print("Output file generated successfully.")

#     # Read the output file
#     with open("hindi_pos.txt", "r", encoding="utf-8") as f:
#         lines = f.readlines()

#     # Count the number of UNK tags
#     unk_count = sum(1 for line in lines if line.strip().split()[1] == "UNK")

#     print("Number of unknown (UNK) tags:", unk_count)

#     import matplotlib.pyplot as plt

#     # Read the output file
#     with open("hindi_pos.txt", "r", encoding="utf-8") as f:
#         lines = f.readlines()

#     # Initialize a dictionary to store tag counts
#     tag_counts = {}

#     # Count the number of each tag
#     for line in lines:
#         _, tag = line.strip().split()
#         tag_counts[tag] = tag_counts.get(tag, 0) + 1

#     # Extract tags and counts for plotting
#     tags = list(tag_counts.keys())
#     counts = list(tag_counts.values())

#     # Plotting
#     plt.figure(figsize=(10, 6))
#     plt.bar(tags, counts, color='skyblue')
#     plt.xlabel('Tags')
#     plt.ylabel('Count')
#     plt.title('Tag Distribution')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()
import os
def random_freq():
    with open(os.path.join('..','..', 'Dataset', 'hindi_pos.txt'), "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Initialize a dictionary to store tag counts
    tag_counts = {}

    # Count the number of each tag
    for line in lines:
        _, tag = line.strip().split()
        tag_counts[tag] = tag_counts.get(tag, 0) + 1

    # Count the total number of words
    total_words = sum(tag_counts.values())

    # Function for randomized most frequent imputation
    def randomized_most_frequent_imputation(word, tag_counts):
        if word == "UNK":
            # Calculate the frequency of each tag
            tag_frequencies = {tag: count / total_words for tag, count in tag_counts.items()}
            # Choose the most frequent tag randomly based on its frequency
            most_frequent_tag = random.choices(list(tag_frequencies.keys()), weights=list(tag_frequencies.values()))[0]
            return most_frequent_tag
        return word

    # Process each line to perform imputation
    output_lines = []
    for line in lines:
        word, tag = line.strip().split()
        tag = randomized_most_frequent_imputation(tag, tag_counts)
        output_lines.append(f"{word} {tag}" if tag != "UNK" else f"{word} {'NN'}")

    # Write the imputed output to a new file
    with open(os.path.join('..', '..','Dataset', "randomized_most_frequent_imputed_output_file.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print("Imputed output file generated successfully.")
def rule_based_implementation():
    with open(os.path.join('..', '..','Dataset', 'hindi_pos.txt'), "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Define a rule-based function to impute UNK tags based on the context or linguistic patterns in Hindi
    def rule_based_imputation(word, previous_word, next_word):
        # Implement your rule-based logic here for Hindi language
            if previous_word and previous_word.endswith("ईं"):
                return "VBG"  # Impute UNK as a gerund (VBG) if the previous word ends with "ईं"
            elif previous_word and (previous_word.endswith("ता") or previous_word.endswith("ती")):
                return "JJ"   # Impute UNK as an adjective (JJ) if the previous word ends with "ता" or "ती"
            elif next_word and next_word.endswith("ने"):
                return "NNP"  # Impute UNK as a proper noun (NNP) if the next word ends with "ने"
            elif next_word and next_word == "संज्ञा":
                return "NN"   # Impute UNK as a noun (NN) if the next word is a noun
            else:
                return "NN"
        

    # Process each line to perform rule-based imputation
    output_lines = []
    for i in range(len(lines)):
        word, tag = lines[i].strip().split()
        previous_word = lines[i-1].split()[0] if i > 0 else None
        next_word = lines[i+1].split()[0] if i < len(lines)-1 else None
        if(tag=="UNK"):
            tag = rule_based_imputation(word, previous_word, next_word)
        output_lines.append(f"{word} {tag}")

    # Write the output with rule-based imputation to a new file
    with open(os.path.join('..', '..','Dataset', "rule_based_imputed_output_file.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print("Rule-based imputed output file generated successfully.")
def most_freq_imputer():
    with open(os.path.join('..', '..','Dataset', 'hindi_pos.txt'), "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Initialize a dictionary to store word counts
    word_counts = {}

    # Count the number of each word
    for line in lines:
        word, tag = line.strip().split()
        word_counts[word] = word_counts.get(word, 0) + 1

    # Count the total number of words
    total_words = sum(word_counts.values())

    # Function for most frequent imputation
    def most_frequent_imputation(word, word_counts):
        if word == "UNK":
            # Calculate the frequency of each word
            word_frequencies = {word: count / total_words for word, count in word_counts.items()}
            # Find the most frequent word
            most_frequent_word = max(word_frequencies, key=word_frequencies.get)
            return most_frequent_word
        return word

    # Process each line to perform imputation
    output_lines = []
    for line in lines:
        word, tag = line.strip().split()
        word = most_frequent_imputation(word, word_counts)
        output_lines.append(f"{word} {tag}" if tag != "UNK" else f"{word} {'NN'}")

    # Write the imputed output to a new file
    with open(os.path.join('..', '..','Dataset', "most_freq_pos.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print("Imputed output file generated successfully.")

if __name__ == "__main__":
    most_freq_imputer()
    

