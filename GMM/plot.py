import matplotlib.pyplot as plt

data = [
    ('Gujrati', 32, 13, 0.0161),
    ('Gujrati', 64, 13, 0.0),
    ('Gujrati', 120, 13, 0.0342),
    ('Gujrati', 32, 24, 0.0164),
    ('Gujrati', 64, 24, 0.0315),
    ('Gujrati', 120, 24, 0.0496),
    ('Gujrati', 32, 39, 1.0),
    ('Gujrati', 64, 39, 0.9804),
    ('Gujrati', 120, 39, 0.995),
    ('Tamil', 32, 13, 0.3557),
    ('Tamil', 64, 13, 0.3672),
    ('Tamil', 120, 13, 0.4167),
    ('Tamil', 32, 24, 0.3643),
    ('Tamil', 64, 24, 0.3835),
    ('Tamil', 120, 24, 0.4101),
    ('Tamil', 32, 39, 0.6014),
    ('Tamil', 64, 39, 0.6014),
    ('Tamil', 120, 39, 0.6755),
    ('Telugu', 32, 13, 0.7085),
    ('Telugu', 64, 13, 0.7091),
    ('Telugu', 120, 13, 0.7397),
    ('Telugu', 32, 24, 0.6909),
    ('Telugu', 64, 24, 0.657),
    ('Telugu', 120, 24, 0.6667),
    ('Telugu', 32, 39, 0.7782),
    ('Telugu', 64, 39, 0.7905),
    ('Telugu', 120, 39, 0.8065),
]

import numpy as np


# Extracting data for each language
languages = ['Gujrati', 'Tamil', 'Telugu']
models = [32, 64, 120]
features = [13, 24, 39]

# Grouping data by language
language_data = {}
for language in languages:
    language_data[language] = {}
    for model in models:
        language_data[language][model] = {}
        for feature in features:
            language_data[language][model][feature] = [d[3] for d in data if d[0] == language and d[1] == model and d[2] == feature]

# Plotting
fig, axs = plt.subplots(1, len(languages), figsize=(15, 5), sharey=True)

for i, language in enumerate(languages):
    ax = axs[i]
    ax.set_ylabel('F1 Score')
    ax.set_xlabel('Model Components')
    ax.set_xticks(np.arange(len(models)) + 0.25)
    ax.set_xticklabels(['32', '64', '120'])
    
    for j, feature in enumerate(features):
        bar_width = 0.2
        index = np.arange(len(models)) + j * bar_width
        
        means = [np.mean(language_data[language][model][feature]) for model in models]
        rects = ax.bar(index, means, bar_width, label=f'{feature} PCA')
        
        for rect, mean in zip(rects, means):
            height = rect.get_height()
            if mean > 0.9:
                ax.annotate(f'{mean:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, -5), textcoords="offset points",
                            ha='center', va='top', rotation=90, color='black')
            else:
                ax.annotate(f'{mean:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', rotation=90, color='black')

plt.tight_layout()
plt.show()
