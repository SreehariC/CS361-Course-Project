def compute_f1_score(confusion_matrix):
    num_classes = len(confusion_matrix)
    f1_scores = []

    for i in range(num_classes):
        true_positive = confusion_matrix[i][i]
        false_positive = sum(confusion_matrix[j][i] for j in range(num_classes) if j != i)
        false_negative = sum(confusion_matrix[i][j] for j in range(num_classes) if j != i)

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1_score)

    return f1_scores

# Example 3x3 confusion matrix where rows are true labels and columns are predicted labels
confusion_matrix_32_24 = [
    [1, 99, 0],
    [9, 47, 44],
    [12, 12, 76]
]

confusion_matrix_32_13 = [
    [1, 99, 0],
    [11, 45, 44],
    [12, 9, 79]
]

confusion_matrix_64_13 = [
    [0, 100, 0],
    [11, 47, 42],
    [13, 9, 78]
]

confusion_matrix_120_13 = [
    [2, 98, 0],
    [7, 55, 38],
    [8, 11, 81]
]

confusion_matrix_120_24 = [
    [3, 97, 0],
    [9, 57, 34],
    [9, 24, 67]
]

confusion_matrix_64_24 = [
    [2, 98, 0],
    [10, 51, 39],
    [15, 17, 68]
]

confusion_matrix_32_39 = [
    [100,0, 0],
    [0, 43, 57],
    [0, 0, 100]
]

confusion_matrix_64_39 = [
    [100, 0, 0],
    [4, 43, 53],
    [0, 0, 100]
]

confusion_matrix_120_39 = [
    [100, 0, 0],
    [1, 51, 48],
    [0, 0, 100]
]

def routine(name,cm):
    print(f"F1 Score for {name}")
    f1_scores = compute_f1_score(cm)

    # Print F1 scores for each class
    for i, f1_score in enumerate(f1_scores):
        print(f"F1 Score for class {i}: {f1_score:.4f}")
    print("="*50)

# Compute F1 score
routine("32_13",confusion_matrix_32_13)
routine("64_13",confusion_matrix_64_13)
routine("120_13",confusion_matrix_120_13)


routine("32_24",confusion_matrix_32_24)
routine("64_24",confusion_matrix_64_24)
routine("120_24",confusion_matrix_120_24)


routine("32_39",confusion_matrix_32_39)
routine("64_39",confusion_matrix_64_39)
routine("120_39",confusion_matrix_120_39)
