
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc

from read_corpus import read_conll_corpus
from feature import FeatureSet, STARTING_LABEL_INDEX

from math import exp, log
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

import time
import random
import json
import datetime

from collections import Counter

SCALING_THRESHOLD = 1e250

ITERATION_NUM = 0
SUB_ITERATION_NUM = 0
TOTAL_SUB_ITERATIONS = 0
GRADIENT = None


def _callback(params):
    global ITERATION_NUM
    global SUB_ITERATION_NUM
    global TOTAL_SUB_ITERATIONS
    ITERATION_NUM += 1
    TOTAL_SUB_ITERATIONS += SUB_ITERATION_NUM
    SUB_ITERATION_NUM = 0

def _generate_potential_table(params, num_labels, feature_set, X, inference=True):
    """
    Generates a potential table using given observations.
    * potential_table[t][prev_y, y]
        := exp(inner_product(params, feature_vector(prev_y, y, X, t)))
        (where 0 <= t < len(X))
    """
    tables = list()
    for t in range(len(X)):
        table = np.zeros((num_labels, num_labels))
        if inference:
            for (prev_y, y), score in feature_set.calc_inner_products(params, X, t):
                if prev_y == -1:
                    table[:, y] += score
                else:
                    table[prev_y, y] += score
        else:
            for (prev_y, y), feature_ids in X[t]:
                score = sum(params[fid] for fid in feature_ids)
                if prev_y == -1:
                    table[:, y] += score
                else:
                    table[prev_y, y] += score
        table = np.exp(table)
        if t == 0:
            table[STARTING_LABEL_INDEX+1:] = 0
        else:
            table[:,STARTING_LABEL_INDEX] = 0
            table[STARTING_LABEL_INDEX,:] = 0
        tables.append(table)

    return tables


def _forward_backward(num_labels, time_length, potential_table):
    """
    Calculates alpha(forward terms), beta(backward terms), and Z(instance-specific normalization factor)
        with a scaling method(suggested by Rabiner, 1989).
    * Reference:
        - 1989, Lawrence R. Rabiner, A Tutorial on Hidden Markov Models and Selected Applications
        in Speech Recognition
    """
    alpha = np.zeros((time_length, num_labels))
    scaling_dic = dict()
    t = 0
    for label_id in range(num_labels):
        alpha[t, label_id] = potential_table[t][STARTING_LABEL_INDEX, label_id]
    #alpha[0, :] = potential_table[0][STARTING_LABEL_INDEX, :]  # slow
    t = 1
    while t < time_length:
        scaling_time = None
        scaling_coefficient = None
        overflow_occured = False
        label_id = 1
        while label_id < num_labels:
            alpha[t, label_id] = np.dot(alpha[t-1,:], potential_table[t][:,label_id])
            if alpha[t, label_id] > SCALING_THRESHOLD:
                if overflow_occured:
                    print('******** Consecutive overflow ********')
                    raise BaseException()
                overflow_occured = True
                scaling_time = t - 1
                scaling_coefficient = SCALING_THRESHOLD
                scaling_dic[scaling_time] = scaling_coefficient
                break
            else:
                label_id += 1
        if overflow_occured:
            alpha[t-1] /= scaling_coefficient
            alpha[t] = 0
        else:
            t += 1

    beta = np.zeros((time_length, num_labels))
    t = time_length - 1
    for label_id in range(num_labels):
        beta[t, label_id] = 1.0
    #beta[time_length - 1, :] = 1.0     # slow
    for t in range(time_length-2, -1, -1):
        for label_id in range(1, num_labels):
            beta[t, label_id] = np.dot(beta[t+1,:], potential_table[t+1][label_id,:])
        if t in scaling_dic.keys():
            beta[t] /= scaling_dic[t]

    Z = sum(alpha[time_length-1])

    return alpha, beta, Z, scaling_dic




def gradient_descnet(params,*args,learning_rate=0.01, num_iterations=100,k=5,patience=10):
    training_data, feature_set, training_feature_data, empirical_counts, label_dic, squared_sigma = args
    
    # Initialize velocity for momentum
    velocity = np.zeros_like(params)
    
    # Initialize variables for early stopping
    best_params = np.copy(params)
    best_val_loss = float('inf')
    no_improvement_count = 0
    
    
    # Split training data into k folds
    fold_size = len(training_feature_data) // k


    for fold in range(k):
        # Split data into training and validation sets for this fold
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size
        
        # Create training and validation datasets
        if fold == k - 1:
            # Use the last part of the data for validation in the last fold to include any remaining samples
            train_data_fold = training_feature_data[:val_start] + training_feature_data[val_end:]
            val_data_fold = training_feature_data[val_start:val_end]
        else:
            train_data_fold = training_feature_data[:val_start] + training_feature_data[val_end:]
            val_data_fold = training_feature_data[val_start:val_end]

            for iteration in range(num_iterations):
            # Compute expected counts and other necessary calculations (omitted for brevity)
            # ...
                expected_counts = np.zeros(len(feature_set))
                #print(len(params))
                total_logZ = 0
                for X_features in training_feature_data:
                    potential_table = _generate_potential_table(params, len(label_dic), feature_set,
                                                                X_features, inference=False)
                    alpha, beta, Z, scaling_dic = _forward_backward(len(label_dic), len(X_features), potential_table)
                    total_logZ += log(Z) + \
                                sum(log(scaling_coefficient) for _, scaling_coefficient in scaling_dic.items())
                    for t in range(len(X_features)):
                        potential = potential_table[t]
                        for (prev_y, y), feature_ids in X_features[t]:
                            # Adds p(prev_y, y | X, t)
                            if prev_y == -1:
                                if t in scaling_dic.keys():
                                    prob = (alpha[t, y] * beta[t, y] * scaling_dic[t])/Z
                                else:
                                    prob = (alpha[t, y] * beta[t, y])/Z
                            elif t == 0:
                                if prev_y is not STARTING_LABEL_INDEX:
                                    continue
                                else:
                                    prob = (potential[STARTING_LABEL_INDEX, y] * beta[t, y])/Z
                            else:
                                if prev_y is STARTING_LABEL_INDEX or y is STARTING_LABEL_INDEX:
                                    continue
                                else:
                                    prob = (alpha[t-1, prev_y] * potential[prev_y, y] * beta[t, y]) / Z
                            for fid in feature_ids:
                                expected_counts[fid] += prob

                likelihood = np.dot(empirical_counts, params) - total_logZ - \
                            np.sum(np.dot(params,params))/(squared_sigma*2)

                gradients = (empirical_counts - expected_counts - params/squared_sigma)
                ## How this comes, why expected counts, because, 1/Z in deno and it changes to prob dist
                global GRADIENT
                GRADIENT = gradients
          
                # Compute gradients
                gradients = _gradient(params)
                
        

                # Update parameters
                params -= learning_rate*gradients
                val_loss = compute_validation_loss(params, val_data_fold, label_dic, feature_set, squared_sigma)
            
                # Check for improvement in validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = np.copy(params)
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                # Early stopping condition
                if no_improvement_count >= patience:
                    print(f"Stopping optimization for fold {fold+1} due to no improvement in validation loss for {patience} iterations.")
                    print(f"Num of iterations {iteration}, likelihood = {likelihood}, val_likelihood={-val_loss}")
                    break
                print(iteration,likelihood)
    return best_params





 



def BFGS(params,*args):
    '''
    DESCRIPTION
    BFGS Quasi-Newton Method, implemented as described in Nocedal:
    Numerical Optimisation.
    INPUTS:
    f:      function to be optimised 
    x0:     intial guess
    max_it: maximum iterations 
    plot:   if the problem is 2 dimensional, returns 
            a trajectory plot of the optimisation scheme.
    OUTPUTS: 
    x:      the optimal solution of the function f 
    '''
    training_data, feature_set, training_feature_data, empirical_counts, label_dic, squared_sigma = args
    iterations=0
    learning_rate=0.01
    H = np.eye(len(feature_set)) # initial hessian
    x = params[:]
    it = 2 
    x_store =  np.array([np.zeros(len(feature_set))])

    while(iterations<100):
       
        expected_counts = np.zeros(len(feature_set))
        #print(len(params))
        total_logZ = 0
        for X_features in training_feature_data:
            potential_table = _generate_potential_table(params, len(label_dic), feature_set,
                                                        X_features, inference=False)
            alpha, beta, Z, scaling_dic = _forward_backward(len(label_dic), len(X_features), potential_table)
            total_logZ += log(Z) + \
                        sum(log(scaling_coefficient) for _, scaling_coefficient in scaling_dic.items())
            for t in range(len(X_features)):
                potential = potential_table[t]
                for (prev_y, y), feature_ids in X_features[t]:
                    # Adds p(prev_y, y | X, t)
                    if prev_y == -1:
                        if t in scaling_dic.keys():
                            prob = (alpha[t, y] * beta[t, y] * scaling_dic[t])/Z
                        else:
                            prob = (alpha[t, y] * beta[t, y])/Z
                    elif t == 0:
                        if prev_y is not STARTING_LABEL_INDEX:
                            continue
                        else:
                            prob = (potential[STARTING_LABEL_INDEX, y] * beta[t, y])/Z
                    else:
                        if prev_y is STARTING_LABEL_INDEX or y is STARTING_LABEL_INDEX:
                            continue
                        else:
                            prob = (alpha[t-1, prev_y] * potential[prev_y, y] * beta[t, y]) / Z
                    for fid in feature_ids:
                        expected_counts[fid] += prob

        likelihood = np.dot(empirical_counts, params) - total_logZ - \
                    np.sum(np.dot(params,params))/(squared_sigma*2)

        gradients = -(empirical_counts - expected_counts - params/squared_sigma)
        global GRADIENT
        GRADIENT = gradients
        
        d = len(feature_set) # dimension of problem 
        nabla = GRADIENT # initial gradient 
        
        it += 1
        p = -H@nabla # search direction (Newton Method)
        a = 0.01 # line search 
        s = a * p 
        x_new = x + a * p 
        params=x_new
        training_data, feature_set, training_feature_data, empirical_counts, label_dic, squared_sigma = args
        expected_counts = np.zeros(len(feature_set))
        #print(len(params))
        total_logZ = 0
        for X_features in training_feature_data:
            potential_table = _generate_potential_table(params, len(label_dic), feature_set,
                                                        X_features, inference=False)
            alpha, beta, Z, scaling_dic = _forward_backward(len(label_dic), len(X_features), potential_table)
            total_logZ += log(Z) + \
                        sum(log(scaling_coefficient) for _, scaling_coefficient in scaling_dic.items())
            for t in range(len(X_features)):
                potential = potential_table[t]
                for (prev_y, y), feature_ids in X_features[t]:
                    # Adds p(prev_y, y | X, t)
                    if prev_y == -1:
                        if t in scaling_dic.keys():
                            prob = (alpha[t, y] * beta[t, y] * scaling_dic[t])/Z
                        else:
                            prob = (alpha[t, y] * beta[t, y])/Z
                    elif t == 0:
                        if prev_y is not STARTING_LABEL_INDEX:
                            continue
                        else:
                            prob = (potential[STARTING_LABEL_INDEX, y] * beta[t, y])/Z
                    else:
                        if prev_y is STARTING_LABEL_INDEX or y is STARTING_LABEL_INDEX:
                            continue
                        else:
                            prob = (alpha[t-1, prev_y] * potential[prev_y, y] * beta[t, y]) / Z
                    for fid in feature_ids:
                        expected_counts[fid] += prob

        likelihood = np.dot(empirical_counts, params) - total_logZ - \
                    np.sum(np.dot(params,params))/(squared_sigma*2)

        gradients = -(empirical_counts - expected_counts - params/squared_sigma)
        
        GRADIENT = gradients

        print(f"Iterations:{iterations}    :    likelihood:{likelihood}")
        iterations+=1
        nabla_new = GRADIENT
        y = nabla_new - nabla 
        y = np.array([y])
        s = np.array([s])
        y = np.reshape(y,(d,1))
        s = np.reshape(s,(d,1))
        r = 1/(y.T@s)
        li = (np.eye(d)-(r*((s@(y.T))))) #updating matrix of H_k left side
        ri = (np.eye(d)-(r*((y@(s.T))))) #updating matrix of H_k right side
        hess_inter = li@H@ri
        H = hess_inter + (r*((s@(s.T)))) # BFGS Update
        nabla = nabla_new[:] 
        x = x_new[:]
        x_store = np.append(x_store, [x_new], axis = 0)
        params=x

import numpy as np

def gradient_descent_with_momentum(params, *args, learning_rate=0.01, momentum=0.9, num_iterations=100,k=5,patience=10):
    """
    Perform gradient descent with momentum optimization and k-fold cross-validation with early stopping.

    Args:
    - params (array-like): Initial parameters (weights or biases).
    - args (tuple): Tuple of arguments needed for optimization (training data, features, etc.).
    - learning_rate (float): Learning rate for gradient descent (default: 0.01).
    - momentum (float): Momentum parameter (default: 0.9).
    - num_iterations (int): Maximum number of iterations for optimization (default: 100).
    - k (int): Number of folds for cross-validation (default: 5).
    - patience (int): Number of iterations with no improvement to wait before stopping (default: 10).

    Returns:
    - optimized_params (array-like): Optimized parameters after gradient descent.
    """
    # Unpack arguments
    iterations = 2
    epsilon = 1e-8
    velocity=np.zeros_like(params)
    iterations=2
    learning_rate=0.1
    while(iterations<400):
        training_data, feature_set, training_feature_data, empirical_counts, label_dic, squared_sigma = args
        expected_counts = np.zeros(len(feature_set))
        #print(len(params))
        total_logZ = 0
        for X_features in training_feature_data:
            potential_table = _generate_potential_table(params, len(label_dic), feature_set,
                                                        X_features, inference=False)
            alpha, beta, Z, scaling_dic = _forward_backward(len(label_dic), len(X_features), potential_table)
            total_logZ += log(Z) + \
                        sum(log(scaling_coefficient) for _, scaling_coefficient in scaling_dic.items())
            for t in range(len(X_features)):
                potential = potential_table[t]
                for (prev_y, y), feature_ids in X_features[t]:
                    # Adds p(prev_y, y | X, t)
                    if prev_y == -1:
                        if t in scaling_dic.keys():
                            prob = (alpha[t, y] * beta[t, y] * scaling_dic[t])/Z
                        else:
                            prob = (alpha[t, y] * beta[t, y])/Z
                    elif t == 0:
                        if prev_y is not STARTING_LABEL_INDEX:
                            continue
                        else:
                            prob = (potential[STARTING_LABEL_INDEX, y] * beta[t, y])/Z
                    else:
                        if prev_y is STARTING_LABEL_INDEX or y is STARTING_LABEL_INDEX:
                            continue
                        else:
                            prob = (alpha[t-1, prev_y] * potential[prev_y, y] * beta[t, y]) / Z
                    for fid in feature_ids:
                        expected_counts[fid] += prob

        likelihood = np.dot(empirical_counts, params) - total_logZ - \
                    np.sum(np.dot(params,params))/(squared_sigma*2)

        gradients = (empirical_counts - expected_counts - params/squared_sigma)
        ## How this comes, why expected counts, because, 1/Z in deno and it changes to prob dist
        global GRADIENT
        GRADIENT = gradients
        
        gradients=_gradient(params)
        velocity = momentum * velocity + learning_rate * gradients

        # Update parameters
        params -= velocity
        
        iterations+=1

def gradient_descent_with_momentum_kfold(params, *args, learning_rate=0.01, momentum=0.9, num_iterations=100,k=5,patience=10):
    """
    Perform gradient descent with momentum optimization and k-fold cross-validation with early stopping.

    Args:
    - params (array-like): Initial parameters (weights or biases).
    - args (tuple): Tuple of arguments needed for optimization (training data, features, etc.).
    - learning_rate (float): Learning rate for gradient descent (default: 0.01).
    - momentum (float): Momentum parameter (default: 0.9).
    - num_iterations (int): Maximum number of iterations for optimization (default: 100).
    - k (int): Number of folds for cross-validation (default: 5).
    - patience (int): Number of iterations with no improvement to wait before stopping (default: 10).

    Returns:
    - optimized_params (array-like): Optimized parameters after gradient descent.
    """
    # Unpack arguments
    training_data, feature_set, training_feature_data, empirical_counts, label_dic, squared_sigma = args
    
    # Initialize velocity for momentum
    velocity = np.zeros_like(params)
    
    # Initialize variables for early stopping
    best_params = np.copy(params)
    best_val_loss = float('inf')
    no_improvement_count = 0
    
    
    # Split training data into k folds
    fold_size = len(training_feature_data) // k


    for fold in range(k):
        # Split data into training and validation sets for this fold
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size
        
        # Create training and validation datasets
        if fold == k - 1:
            # Use the last part of the data for validation in the last fold to include any remaining samples
            train_data_fold = training_feature_data[:val_start] + training_feature_data[val_end:]
            val_data_fold = training_feature_data[val_start:val_end]
        else:
            train_data_fold = training_feature_data[:val_start] + training_feature_data[val_end:]
            val_data_fold = training_feature_data[val_start:val_end]

            for iteration in range(num_iterations):
            # Compute expected counts and other necessary calculations (omitted for brevity)
            # ...
                expected_counts = np.zeros(len(feature_set))
                #print(len(params))
                total_logZ = 0
                for X_features in training_feature_data:
                    potential_table = _generate_potential_table(params, len(label_dic), feature_set,
                                                                X_features, inference=False)
                    alpha, beta, Z, scaling_dic = _forward_backward(len(label_dic), len(X_features), potential_table)
                    total_logZ += log(Z) + \
                                sum(log(scaling_coefficient) for _, scaling_coefficient in scaling_dic.items())
                    for t in range(len(X_features)):
                        potential = potential_table[t]
                        for (prev_y, y), feature_ids in X_features[t]:
                            # Adds p(prev_y, y | X, t)
                            if prev_y == -1:
                                if t in scaling_dic.keys():
                                    prob = (alpha[t, y] * beta[t, y] * scaling_dic[t])/Z
                                else:
                                    prob = (alpha[t, y] * beta[t, y])/Z
                            elif t == 0:
                                if prev_y is not STARTING_LABEL_INDEX:
                                    continue
                                else:
                                    prob = (potential[STARTING_LABEL_INDEX, y] * beta[t, y])/Z
                            else:
                                if prev_y is STARTING_LABEL_INDEX or y is STARTING_LABEL_INDEX:
                                    continue
                                else:
                                    prob = (alpha[t-1, prev_y] * potential[prev_y, y] * beta[t, y]) / Z
                            for fid in feature_ids:
                                expected_counts[fid] += prob

                likelihood = np.dot(empirical_counts, params) - total_logZ - \
                            np.sum(np.dot(params,params))/(squared_sigma*2)

                gradients = (empirical_counts - expected_counts - params/squared_sigma)
                ## How this comes, why expected counts, because, 1/Z in deno and it changes to prob dist
                global GRADIENT
                GRADIENT = gradients
                
                gradients=_gradient(params)
                velocity = momentum * velocity + learning_rate * gradients

                # Compute gradients
                gradients = _gradient(params)
                
                # Update velocity using momentum
                velocity = momentum * velocity + learning_rate * gradients

                # Update parameters
                params -= velocity
                val_loss = compute_validation_loss(params, val_data_fold, label_dic, feature_set, squared_sigma)
            
                # Check for improvement in validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = np.copy(params)
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                # Early stopping condition
                if no_improvement_count >= patience:
                    print(f"Stopping optimization for fold {fold+1} due to no improvement in validation loss for {patience} iterations.")
                    print(f"Num of iterations {iteration}, likelihood = {likelihood}, val_likelihood={-val_loss}")
                    break
                print(f"Iterations : { iteration} , Likelihood : {likelihood}")
    return best_params

def compute_empirical_counts(training_feature_data, feature_set, label_dic):
    """
    Compute empirical feature counts based on the training data.

    Args:
    - training_feature_data (list): List of training feature data, where each element represents a sequence of feature occurrences.
    - feature_set (list): List of unique feature identifiers.
    - label_dic (dict): Dictionary mapping label names to indices.

    Returns:
    - empirical_counts (array-like): Array of empirical counts for each feature.
    """

    num_features = len(feature_set)
    num_labels = len(label_dic)
    empirical_counts = np.zeros(num_features)

    for sequence in training_feature_data:
        for t in range(len(sequence)):
            for (prev_y, y), feature_ids in sequence[t]:
                # Increment counts for each feature associated with the feature_ids
                for fid in feature_ids:
                    #print(len(empirical_counts))
                    empirical_counts[fid] += 1

    return empirical_counts

def compute_validation_loss(params, val_data_fold, label_dic, feature_set, squared_sigma):
    """
    Compute validation loss for given parameters on validation data within a fold.

    Args:
    - params (array-like): Parameters (weights or biases) to evaluate.
    - val_data_fold (list): Validation feature data for the current fold.
    - label_dic (dict): Dictionary mapping label names to indices.
    - feature_set (list): List of feature names.
    - squared_sigma (float): Squared sigma value for regularization.

    Returns:
    - val_loss (float): Validation loss computed for the given parameters.
    """
    total_logZ = 0
    expected_counts = np.zeros(len(feature_set))
    
    for X_features in val_data_fold:
        potential_table = _generate_potential_table(params, len(label_dic), feature_set, X_features, inference=False)
        alpha, beta, Z, scaling_dic = _forward_backward(len(label_dic), len(X_features), potential_table)
        total_logZ += np.log(Z) + sum(np.log(scaling_coefficient) for _, scaling_coefficient in scaling_dic.items())
        
        for t in range(len(X_features)):
            potential = potential_table[t]
            for (prev_y, y), feature_ids in X_features[t]:
                # Adds p(prev_y, y | X, t)
                if prev_y == -1:
                    if t in scaling_dic.keys():
                        prob = (alpha[t, y] * beta[t, y] * scaling_dic[t])/Z
                    else:
                        prob = (alpha[t, y] * beta[t, y])/Z
                elif t == 0:
                    if prev_y is not STARTING_LABEL_INDEX:
                        continue
                    else:
                        prob = (potential[STARTING_LABEL_INDEX, y] * beta[t, y])/Z
                else:
                    if prev_y is STARTING_LABEL_INDEX or y is STARTING_LABEL_INDEX:
                        continue
                    else:
                        prob = (alpha[t-1, prev_y] * potential[prev_y, y] * beta[t, y]) / Z
                for fid in feature_ids:
                    expected_counts[fid] += prob

    
    empirical_counts = compute_empirical_counts(val_data_fold, feature_set, label_dic)
    likelihood = np.dot(empirical_counts, params) - total_logZ - np.sum(params**2) / (2 * squared_sigma)
    gradients = empirical_counts - expected_counts - params / squared_sigma
    
    return -likelihood







def _gradient(params, *args):
    return GRADIENT * -1
train_A=[]
test_A=[]
class LinearChainCRF():
    """
    Linear-chain Conditional Random Field
    """

    training_data = None
    feature_set = None

    label_dic = None
    label_array = None
    num_labels = None

    params = None

    # For L-BFGS
    squared_sigma = 10.0

    def __init__(self):
        pass

    def _read_corpus(self, filename):
        return read_conll_corpus(filename)

    def _get_training_feature_data(self):
        return [[self.feature_set.get_feature_list(X, t) for t in range(len(X))]
                for X in self.training_data]
    



    

    def _estimate_parameters(self):
        """
        Estimates parameters using L-BFGS.
        * References:
            - R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound Constrained Optimization,
            (1995), SIAM Journal on Scientific and Statistical Computing, 16, 5, pp. 1190-1208.
            - C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B, FORTRAN routines for large
            scale bound constrained optimization (1997), ACM Transactions on Mathematical Software, 23, 4,
            pp. 550 - 560.
            - J.L. Morales and J. Nocedal. L-BFGS-B: Remark on Algorithm 778: L-BFGS-B, FORTRAN routines for
            large scale bound constrained optimization (2011), ACM Transactions on Mathematical Software, 38, 1.
        """
        training_feature_data = self._get_training_feature_data()
        print('* Squared sigma:', self.squared_sigma)
        print('* Start Training')
        print('   ========================')
        
       # print(self.feature_set)
        self.params=np.random.rand(len(self.feature_set))
        # self.params, log_likelihood, information = \
        #         fmin_l_bfgs_b(func=_log_likelihood, fprime=_gradient,
        #                       x0=np.zeros(len(self.feature_set)),
        #                       args=(self.training_data, self.feature_set, training_feature_data,
        #                             self.feature_set.get_empirical_counts(),
        #                             self.label_dic, self.squared_sigma),
        #                       callback=_callback)
        #gradient_descent_with_adam(self.params,self.training_data, self.feature_set, training_feature_data,
                                    ###self.label_dic, self.squared_sigma)
        # BFGS(self.params,self.training_data, self.feature_set, training_feature_data,
        #                              self.feature_set.get_empirical_counts(),
        #                              self.label_dic, self.squared_sigma)
        gradient_descent_with_momentum_kfold(self.params,self.training_data, self.feature_set, training_feature_data,self.feature_set.get_empirical_counts(),
                                         self.label_dic, self.squared_sigma,learning_rate=0.01,momentum=0.9,num_iterations=300)
        # gradient_descnet(self.params,self.training_data, self.feature_set, training_feature_data,
        #                               self.feature_set.get_empirical_counts(),
        #                               self.label_dic, self.squared_sigma,num_iterations=300)
        print('   ========================')
        print('   (iter: iteration, sit: sub iteration)')


    def train(self, corpus_filename):
        """
        Estimates parameters using conjugate gradient methods.(L-BFGS-B used)
        """
        start_time = time.time()
        print('[%s] Start training' % datetime.datetime.now())

        # Read the training corpus
        print("* Reading training data ... ", end="")
        self.training_data = self._read_corpus(corpus_filename)
        print("Done")
        for i in range(len(self.training_data)):
            if(random.random()>0.1):
                train_A.append(self.training_data[i])
            else:
                test_A.append(self.training_data[i])

        # Generate feature set from the corpus
        self.training_data=train_A
        self.feature_set = FeatureSet()
        self.feature_set.scan(self.training_data)
        self.label_dic, self.label_array = self.feature_set.get_labels()
        self.num_labels = len(self.label_array)
        print("* Number of labels: %d" % (self.num_labels-1))
        print("* Number of features: %d" % len(self.feature_set))

        # Estimates parameters to maximize log-likelihood of the corpus.
        self._estimate_parameters()

        #self.save_model(model_filename)

        elapsed_time = time.time() - start_time
        print('* Elapsed time: %f' % elapsed_time)
        print('* [%s] Training done' % datetime.datetime.now())

    def test(self):
        if self.params is None:
            raise BaseException("You should load a model first!")

        test_data = test_A
        true_labels=[]
        pred_labels=[]
        total_count = 0
        correct_count = 0
        for X in test_data:
            Yprime = self.inference(X)
            for t in range(len(X)):
                true_labels.append(X[t][2])
                pred_labels.append(Yprime[t])
                total_count += 1
                if X[t][2] == Yprime[t]:
                    correct_count += 1
        report = classification_report(true_labels, pred_labels, output_dict=True,zero_division=0)

        # Extract class labels and F1 scores
        class_labels = list(report.keys())[:-3]
        F1_scores = [report[label]['f1-score'] for label in class_labels]
        mean_f1_score = np.mean(F1_scores)
        print("Mean F1 score: ",mean_f1_score)
        print('Correct: %d' % correct_count)
        print('Total: %d' % total_count)
        print('Performance: %f' % (correct_count/total_count))
        # 1. Confusion Matrix
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

        # 2. Precision-Recall Curve
        # precision, recall, _ = precision_recall_curve(true_labels, pred_labels)
        # plt.figure(figsize=(8, 6))
        # plt.plot(recall, precision, color='b', alpha=0.8)
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.title('Precision-Recall Curve')
        # plt.show()

        # 3. ROC Curve
        # fpr, tpr, _ = roc_curve(true_labels, pred_labels)
        # roc_auc = auc(fpr, tpr)
        # plt.figure(figsize=(8, 6))
        # plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic (ROC) Curve')
        # plt.legend(loc='lower right')
        # plt.show()

        # 4. Class-wise F1 Scores
        plt.figure(figsize=(10, 6))
        plt.bar(class_labels, F1_scores, color='skyblue')
        plt.xlabel('Class Labels')
        plt.ylabel('F1 Score')
        plt.title('Class-wise F1 Scores')
        plt.xticks(rotation=45)
        plt.show()

        # 5. Overall Performance Metrics
        print("Mean F1 score:", mean_f1_score)
        print("Accuracy:", correct_count / total_count)

    def print_test_result(self, test_corpus_filename):
        test_data = self._read_corpus(test_corpus_filename)

        for X, Y in test_data:
            Yprime = self.inference(X)
            for t in range(len(X)):
                print('%s\t%s\t%s' % ('\t'.join(X[t]), Y[t], Yprime[t]))
            print()

    def inference(self, X):
        """
        Finds the best label sequence.
        """
        potential_table = _generate_potential_table(self.params, self.num_labels,
                                                    self.feature_set, X, inference=True)
        Yprime = self.viterbi(X, potential_table)
        return Yprime

    def viterbi(self, X, potential_table):
        """
        The Viterbi algorithm with backpointers
        """
        time_length = len(X)
        max_table = np.zeros((time_length, self.num_labels))
        argmax_table = np.zeros((time_length, self.num_labels), dtype='int64')

        t = 0
        for label_id in range(self.num_labels):
            max_table[t, label_id] = potential_table[t][STARTING_LABEL_INDEX, label_id]
        for t in range(1, time_length):
            for label_id in range(1, self.num_labels):
                max_value = -float('inf')
                max_label_id = None
                for prev_label_id in range(1, self.num_labels):
                    value = max_table[t-1, prev_label_id] * potential_table[t][prev_label_id, label_id]
                    if value > max_value:
                        max_value = value
                        max_label_id = prev_label_id
                max_table[t, label_id] = max_value
                argmax_table[t, label_id] = max_label_id

        sequence = list()
        next_label = max_table[time_length-1].argmax()
        sequence.append(next_label)
        for t in range(time_length-1, -1, -1):
            next_label = argmax_table[t, next_label]
            sequence.append(next_label)
        labels=[]
        for i in sequence[::-1]:
            for key,value in self.label_dic.items():
                if value==i:
                    labels.append(key)
        return labels[1:]


    def save_model(self, model_filename):
        model = {"feature_dic": self.feature_set.serialize_feature_dic(),
                 "num_features": self.feature_set.num_features,
                 "labels": self.feature_set.label_array,
                 "params": list(self.params)}
        f = open(model_filename, 'w')
        json.dump(model, f, ensure_ascii=False, indent=2, separators=(',', ':'))
        f.close()
        import os
        print('* Trained CRF Model has been saved at "%s/%s"' % (os.getcwd(), model_filename))

    def load(self, model_filename):
        f = open(model_filename)
        model = json.load(f)
        f.close()

        self.feature_set = FeatureSet()
        self.feature_set.load(model['feature_dic'], model['num_features'], model['labels'])
        self.label_dic, self.label_array = self.feature_set.get_labels()
        self.num_labels = len(self.label_array)
        self.params = np.asarray(model['params'])

        print('CRF model loaded')


crf = LinearChainCRF()
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
    cnt=0
    with open(filename, 'w', encoding='utf-8') as file:
            for item in data:
                    if(cnt>=500):return
                    if(item[0]=='0'):cnt+=1
                    file.write(item + '\n')

# Function to read data from a file
def read_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = [line.strip().split() for line in file.readlines()]
    return data

input_filename = 'most_freq_pos.txt'
data = read_from_file(input_filename)
# Convert data
converted_data = convert_data(data)

# Write converted data to an output file
output_filename = 'hindi_train1.txt'
write_to_file(converted_data, output_filename)
crf.train('hindi_train1.txt')
crf.test()


