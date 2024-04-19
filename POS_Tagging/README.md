# crf

A pure python implementation of the Linear-Chain Conditional Random Fields

## Dependencies

- `numpy`
- `scipy`
- `matplotlib`
- `scikitlearn`
- `sklearn_crfsuite` (for running library implementation)
## Usage

To run the code, do :
```
python crf.py
```
- The Data is used is `most_freq_pos.txt` from the Dataset Folder and is divided into Train,Test and Val in this code.

To change features taken into account during training, go to `default_feature_fun`c in `feature.py` and desirable features as done there.

# HMM for POS Tagging

## Dependencies
- `numpy`
- `scipy`
- `matplotlib`
- `scikitlearn`
- `HMMLearn` (for running library implementation, for custom not required)
- `nltk` (for running library implementation, for custom not required)

### Usage


To run the code, do :
```
python HMM.py
```
The Data is used is `most_freq_pos.txt` from the Dataset Folder and is divided into Train,Test and Val in this code.

To run the preprocessing part, do :
```
python Preprocessing.py
```

In this, call the method(way you want to impute, there are 3 functions corresponding to three different ways, with clear naming done for each) you want to use to impute the missing values.

The Data is used is `hindi_pos.txt` from the Dataset Folder and imputed output is generated in `Dataset` Folder itself


