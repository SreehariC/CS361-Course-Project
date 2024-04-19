
from collections import Counter
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
import numpy as np


STARTING_LABEL = '*'        # Label of t=-1
STARTING_LABEL_INDEX = 0

def default_feature_func(sent,i):
    word = sent[i][1]

    features = []

    
    features.append('WORD:%s' % word)
    
    # Add context features
    if i > 0:
        word1 = sent[i - 1][1]
        features.append('PREV_WORD:%s' % word1)
        # #features.append('PREV_len(word):%d' % len(word1))
    #     features.append('PREV_word[:3]:%s' % word1[:3])
    #     features.append('PREV_word[:2]:%s' % word1[:2])
    #     features.append('PREV_word[-3:]:%s' % word1[-3:])
    #     features.append('PREV_word[-2:]:%s' % word1[-2:])
    else:
        features.append('BOS:True')
    # if i>1:
    #     word2 = sent[i - 2][1]
    #     features.append('-2:word:%s' % word2)
    # #     features.append('-2:len(word):%d' % len(word2))
    # # #     features.append('-2:word.lower():%s' % word2.lower())
    #     features.append('-2:word[:3]:%s' % word2[:3])
    #     features.append('-2:word[:2]:%s' % word2[:2])
    #     features.append('-2:word[-3:]:%s' % word2[-3:])
    #     features.append('-2:word[-2:]:%s' % word2[-2:])
    # if i < len(sent) - 1:
    #     word1 = sent[i + 1][1]
    #     features.append('NEXT_WORD:%s' % word1)
    # # #     features.append('NEXT_len(word):%d' % len(word1))
    #     features.append('NEXT_word[:3]:%s' % word1[:3])
    #     features.append('NEXT_word[:2]:%s' % word1[:2])
    #     features.append('NEXT_word[-3:]:%s' % word1[-3:])
    #     features.append('NEXT_word[-2:]:%s' % word1[-2:])
    # else:
    #     features.append('EOS:True')

    # if i < len(sent) - 2:
    #     word1 = sent[i + 2][1]
    #     features.append('NEXT_NEXT_WORD:%s' % word1)
    # # # #     features.append('NEXT_NEXT_len(word):%d' % len(word1))
    #     features.append('NEXT_NEXT_word[:3]:%s' % word1[:3])
    #     features.append('NEXT_NEXT_word[:2]:%s' % word1[:2])
    #     features.append('NEXT_NEXT_word[-3:]:%s' % word1[-3:])
    #     features.append('NEXT_NEXT_word[-2:]:%s' % word1[-2:])
    return features





class FeatureSet():
    feature_dic = dict()
    observation_set = set()
    empirical_counts = Counter()
    num_features = 0

    label_dic = {STARTING_LABEL: STARTING_LABEL_INDEX}
    label_array = [STARTING_LABEL]

    feature_func = default_feature_func

    def __init__(self, feature_func=None):
        # Sets a custom feature function.
        if feature_func is not None:
            self.feature_func = feature_func

    def scan(self, data):
        """
        Constructs a feature set, a label set,
            and a counter of empirical counts of each feature from the input data.
        :param data: A list of (X, Y) pairs. (X: observation vector , Y: label vector)
        """
        # Constructs a feature set, and counts empirical counts.
        for X in data:
            prev_y = STARTING_LABEL_INDEX
            for t in range(len(X)):
                # Gets a label id
                try:
                    y = self.label_dic[X[t][2]]
                except KeyError:
                    y = len(self.label_dic)
                    self.label_dic[X[t][2]] = y
                    self.label_array.append(X[t][2])
                # Adds features
                self._add(prev_y, y, X, t)
                prev_y = y

    def load(self, feature_dic, num_features, label_array):
        self.num_features = num_features
        self.label_array = label_array
        self.label_dic = {label: i for label, i in enumerate(label_array)}
        self.feature_dic = self.deserialize_feature_dic(feature_dic)

    def __len__(self):
        return self.num_features

    def _add(self, prev_y, y, X, t):
        """
        Generates features, constructs feature_dic.
        :param prev_y: previous label
        :param y: present label
        :param X: observation vector
        :param t: time
        """

        for feature_string in default_feature_func(X, t):
            # print(feature_string)
            if feature_string in self.feature_dic.keys():
                if (prev_y, y) in self.feature_dic[feature_string].keys():
                    self.empirical_counts[self.feature_dic[feature_string][(prev_y, y)]] += 1
                else:
                    feature_id = self.num_features
                    self.feature_dic[feature_string][(prev_y, y)] = feature_id
                    self.empirical_counts[feature_id] += 1
                    self.num_features += 1
                if (-1, y) in self.feature_dic[feature_string].keys():
                    self.empirical_counts[self.feature_dic[feature_string][(-1, y)]] += 1
                else:
                    feature_id = self.num_features
                    self.feature_dic[feature_string][(-1, y)] = feature_id
                    self.empirical_counts[feature_id] += 1
                    self.num_features += 1
            else:
                self.feature_dic[feature_string] = dict()
                # Bigram feature
                feature_id = self.num_features
                self.feature_dic[feature_string][(prev_y, y)] = feature_id
                self.empirical_counts[feature_id] += 1
                self.num_features += 1
                # Unigram feature
                feature_id = self.num_features
                self.feature_dic[feature_string][(-1, y)] = feature_id
                self.empirical_counts[feature_id] += 1
                self.num_features += 1

    def get_feature_vector(self, prev_y, y, X, t):
        """
        Returns a list of feature ids of given observation and transition.
        :param prev_y: previous label
        :param y: present label
        :param X: observation vector
        :param t: time
        :return: A list of feature ids
        """
        feature_ids = list()
        for feature_string in self.feature_func(X, t):
            print(len(feature_string))
            try:
                feature_ids.append(self.feature_dic[feature_string][(prev_y, y)])
            except KeyError:
                pass
        #print(feature_ids)
        return feature_ids

    def get_labels(self):
        """
        Returns a label dictionary and array.
        """
        return self.label_dic, self.label_array

    def calc_inner_products(self, params, X, t):
        """
        Calculates inner products of the given parameters and feature vectors of the given observations at time t.
        :param params: parameter vector
        :param X: observation vector
        :param t: time
        :return:
        """
        inner_products = Counter()
        for feature_string in default_feature_func(X, t):
            try:
                for (prev_y, y), feature_id in self.feature_dic[feature_string].items():
                    inner_products[(prev_y, y)] += params[feature_id]
            except KeyError:
                pass
        return [((prev_y, y), score) for (prev_y, y), score in inner_products.items()]

    def get_empirical_counts(self):
        empirical_counts = np.ndarray((self.num_features,))
        for feature_id, counts in self.empirical_counts.items():
            empirical_counts[feature_id] = counts
        return empirical_counts

    def get_feature_list(self, X, t):
        feature_list_dic = dict()
        for feature_string in default_feature_func(X, t):
            for (prev_y, y), feature_id in self.feature_dic[feature_string].items():
                if (prev_y, y) in feature_list_dic.keys():
                    feature_list_dic[(prev_y, y)].add(feature_id)
                else:
                    feature_list_dic[(prev_y, y)] = {feature_id}
        return [((prev_y, y), feature_ids) for (prev_y, y), feature_ids in feature_list_dic.items()]
