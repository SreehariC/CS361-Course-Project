import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc
import os
from sys import stdout
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


np.set_printoptions(threshold=np.inf)

#Hyperparameters
number_of_states = 4
number_of_gaussians = 3
sound_path='../Dataset/HindiDigits/'



# Constants
cov_bias = 0.001
cov_bias_init = 0.1
min_log_acc = -1500
log_offset = -700
scale_factor = 10
normalize = True


class GMM_HMM:

    def __init__(self, name, n_states, n_mixure_count):
        self.name = name
        self.states_count = n_states
        self.mixture_count = n_mixure_count

        # random transaction probabilities
        self.A = np.ones((n_states, n_states)) / n_states

        # random prior
        self.prior = np.ones(n_states) / n_states

        # uniform mixure coefficients
        self.c = np.ones((n_states, n_mixure_count)) / n_mixure_count

        # input dim
        self.dim_count = None

        # gmm means
        self.mu = None

        # gmm covariances
        self.cov = None

    # ********************************************************************************************

    def init_gmm(self, dataset):

        # set input dimentions
        self.dim_count = dataset[0].shape[1]

        # init mu for gmm
        self.mu = np.random.rand(self.states_count, self.mixture_count, self.dim_count)

        for i in range(self.states_count):
                for j in range(self.mixture_count):
                    obs_idx = np.random.choice(np.arange(len(dataset)))
                    time = np.random.choice(np.arange(dataset[obs_idx].shape[0]))
                    self.mu[i][j] = dataset[obs_idx][time]

        # init cov matrix
        self.cov = np.random.rand(self.states_count, self.mixture_count, self.dim_count, self.dim_count)


        for i in range(self.states_count):
                for j in range(self.mixture_count):
                    obs_idx = np.random.choice(np.arange(len(dataset)))
                    time = np.random.choice(np.arange(dataset[obs_idx].shape[0]))
                    obs = dataset[obs_idx][time].reshape(-1, 1)
                    self.cov[i][j] = np.diag(np.diag(np.dot(obs, obs.T)) + cov_bias_init)

    # ********************************************************************************************

    def forward_pass(self, observation_prob):
        """
        Computes the forward probabilities for the given observation sequence
        in the Hidden Markov Model (HMM).

        Args:
            observation_prob (numpy.ndarray): A 2D array containing the log probabilities
                of observing the given observation sequence for each state at each time step.

        Returns:
            numpy.ndarray: A 2D array containing the forward probabilities for
                each state at each time step.
        """
        T = observation_prob.shape[1]  # Number of time steps
        
        # Initialize forward probabilities
        log_alpha = np.full((self.states_count, T), float('-inf'))  # Initialize with a very small value

        # Calculate the log probability for the first time step
        for i in range(self.states_count):
            self.prior[i] += np.exp(log_offset)
        log_alpha[:, 0] = np.log(self.prior) + observation_prob[:, 0]

        # Calculate the forward probabilities for the remaining time steps
        for t in range(1, T):
            for j in range(self.states_count):
                for i in range(self.states_count):
                    log_transition = log_alpha[i, t - 1]
                    log_transition += np.log(self.A[i, j] + np.exp(log_offset))
                    log_transition += observation_prob[j, t]
                    log_alpha[j, t] = np.logaddexp(log_alpha[j, t], log_transition)

        return log_alpha

    # ********************************************************************************************

    def backward_pass(self, observation_prob):
        """
        Computes the backward probabilities for the given observation sequence
        in the Hidden Markov Model (HMM).

        Args:
            observation_prob (numpy.ndarray): A 2D array containing the log probabilities
                of observing the given observation sequence for each state at each time step.

        Returns:
            numpy.ndarray: A 2D array containing the backward probabilities for
                each state at each time step.
        """
        T = observation_prob.shape[1]  # Number of time steps

        # Initialize backward probabilities
        log_beta = np.full((self.states_count, T), float('-inf'))  # Initialize with a very small value

        # Set the log probabilities for the last time step to 0
        log_beta[:, T - 1] = 0

        # Calculate the backward probabilities for the remaining time steps
        for t in range(T - 2, -1, -1):
            for i in range(self.states_count):
                for j in range(self.states_count):
                    log_transition = np.log(self.A[i, j] + np.exp(log_offset))
                    log_transition += observation_prob[j, t + 1]
                    log_transition += log_beta[j, t + 1]
                    log_beta[i, t] = np.logaddexp(log_beta[i, t], log_transition)

        return log_beta

    # ********************************************************************************************


    def calculate_observation_probability(self, observation):
        """
        Calculates the probability of observing the given observation sequence
        for each state in the Hidden Markov Model (HMM).

        Args:
            observation (numpy.ndarray): A sequence of observations.

        Returns:
            numpy.ndarray: A 2D array containing the log probabilities of observing
                the given observation sequence for each state at each time step.
        """
        T = observation.shape[0]
        observation_prob = np.zeros((self.states_count, T))

        for i in range(self.states_count):
            for t in range(T):
                log_prob = float('-inf')  # Initialize with a very small value

                for j in range(self.mixture_count):
                    # Calculate the Cholesky decomposition of the covariance matrix
                    chol_cov = np.linalg.cholesky(self.cov[i, j])

                    # Calculate the log probability for the current mixture component
                    log_mix_prob = np.log(self.c[i, j] + np.exp(log_offset))
                    log_mix_prob += -np.log((2 * np.pi) ** (self.dim_count / 2))
                    log_mix_prob += -np.log(np.linalg.det(chol_cov) ** 2 + np.exp(log_offset))
                    log_mix_prob += -0.5 * np.dot((observation[t] - self.mu[i, j]).T,
                                                np.dot(np.linalg.inv(self.cov[i, j]),
                                                        (observation[t] - self.mu[i, j])))

                    # Update the log probability for the current state and time step
                    log_prob = np.logaddexp(log_prob, log_mix_prob)

                observation_prob[i, t] = log_prob

        return observation_prob

    # ********************************************************************************************

    def train(self, dataset, stop_diff):

        print('\n--- Running Training for module "{}" '.format(self.name))

        #Normalize the data
        if normalize:
            for i in range(len(dataset)):
                dataset[i] = (dataset[i] - np.mean(dataset[i], axis=0)) / np.std(dataset[i], axis=0)

        self.init_gmm(dataset)

        counter = 1

        current_liklihood = 0.001
        accum_liklihood_prev = 0
        
        #Data Covariance Matrix (dXd)
        data_cov = np.zeros((self.dim_count, self.dim_count))
        for observation in dataset:
            data_cov += np.einsum('ij,ik->jk', observation, observation)

        while True:

            accum_liklihood_prev = current_liklihood
            current_liklihood = 0

            PriorUpdate = np.zeros(self.states_count)
            A_New = np.zeros((self.states_count, self.states_count))
            Mu_New = np.zeros(shape=self.mu.shape)
            C_New = np.zeros(shape=self.c.shape)
            Cov_New = np.zeros(shape=self.cov.shape)

            for observation in dataset:

                stdout.write(
                    '\r---------- Iteration : {}'.format(counter))
                stdout.flush()

                T = observation.shape[0]
                obs_prob = self.calculate_observation_probability(observation)

                alpha = self.forward_pass(obs_prob)
                current_liklihood += np.sum(np.exp(alpha), axis=0)[-1]

                beta = self.backward_pass(obs_prob)
                epsilon = np.zeros((self.states_count, self.states_count, T - 1))

                for t in range(T - 1):
                    accum = np.logaddexp.reduce(epsilon[:,:, t], initial=min_log_acc)
                    epsilon[:, :, t] -= accum

                gamma = np.zeros((self.states_count, T))
                for t in range(T):
                    gamma[:, t] = alpha[:, t] + beta[:, t]
                    gamma_sum = np.logaddexp.reduce(gamma[:, t], initial=min_log_acc)
                    gamma[:, t] = gamma[:, t] - gamma_sum

                h = np.zeros((self.states_count, self.mixture_count, T))
                for t in range(T):
                    for i in range(self.states_count):
                        for j in range(self.mixture_count):

                            if obs_prob[i, t] != -np.Inf:

                                chel = np.linalg.cholesky(self.cov[i, j])

                                new_prob = np.log(self.c[i, j] + np.exp(log_offset))
                                new_prob += - np.log((2 * np.pi) ** (self.dim_count / 2))
                                new_prob += - np.log(np.linalg.det(chel) ** 2 + np.exp(log_offset))
                                new_prob += - 0.5 * np.dot((observation[t] - self.mu[i, j]).T,
                                                           np.dot(np.linalg.inv(self.cov[i, j]),
                                                                  (observation[t] - self.mu[i, j])))

                                h[i, j, t] = new_prob
                                h[i, j, t] -= obs_prob[i, t]

                            else:
                                h[i, j, t] = -np.Inf

                temp_A_update = np.zeros((self.states_count, self.states_count))
                for i in range(self.states_count):
                    for j in range(self.states_count):

                        eps_sum = np.logaddexp.reduce(epsilon[i, :, :T-1],initial=min_log_acc)     
                        # Calculate the accumulation of epsilon values for the specific state transition
                        accum = np.logaddexp.reduce(epsilon[i, j, :T-1],initial=min_log_acc)
                        # Update temp_A_update with the difference
                        temp_A_update[i, j] = accum.item() - np.sum(eps_sum).item()

                temp_A_update = np.exp(temp_A_update)

                average = np.zeros((self.states_count, self.mixture_count))
                for i in range(self.states_count):
                    for j in range(self.mixture_count):
                        average[i, j] = np.sum(np.exp(gamma[i] + h[i, j]))

                temp_mu_update = np.zeros(shape=self.mu.shape)
                for i in range(self.states_count):
                    for j in range(self.mixture_count):
                        accum = 0
                        for t in range(T):
                            accum += np.exp(gamma[i, t] + h[i, j, t]) * observation[t]
                        temp = average[i, j]
                        temp += (average[i, j] == 0)
                        temp_mu_update[i, j] = accum / temp
                        # Mu_New[i, j] = np.dot(gamma[i] * h[i, j], observation) / average[i, j]

                temp_c_update = np.zeros(shape=self.c.shape)
                for i in range(self.states_count):
                    sum = np.sum(average[i])
                    sum += (sum == 0)
                    temp_c_update[i, :] = average[i] / sum

                temp_cov_update = np.zeros_like(self.cov)

                for i in range(self.states_count):
                    for j in range(self.mixture_count):
                        # Initialize accumulator
                        accum = np.zeros((self.dim_count, self.dim_count))
                        
                        for t in range(T):
                            # Calculate difference between observation[t] and temp_mu_update[i, j]
                            diff = (observation[t] - temp_mu_update[i, j]).reshape(-1, 1)
                            
                            # Calculate contribution to the covariance update
                            contribution = np.exp(gamma[i, t] + h[i, j, t]) * np.dot(diff, diff.T)
                            
                            # Accumulate contributions
                            accum += contribution
                        
                        # Ensure average[i, j] is not zero to avoid division by zero
                        average[i, j] += (average[i, j] == 0)
                        
                        # Calculate the covariance update
                        temp_cov_update[i, j] = (accum / average[i, j]) + (cov_bias * np.eye(self.dim_count))

                PriorUpdate += np.exp(gamma[:, 0])
                A_New += temp_A_update
                Mu_New += temp_mu_update
                C_New += temp_c_update
                Cov_New += temp_cov_update

            self.prior = PriorUpdate / len(dataset)
            self.A = A_New / len(dataset)
            self.mu = Mu_New / len(dataset)
            self.c = C_New / len(dataset)
            self.cov = Cov_New / len(dataset)

            counter += 1

            # prevent devide by zero
            current_liklihood += current_liklihood == 0
            if np.abs((accum_liklihood_prev - current_liklihood) / current_liklihood) < stop_diff:
                return


    def likelihood(self, dataset):
        # Check if normalization is needed
        if normalize:
            # Normalize each observation in the dataset
            for i in range(len(dataset)):
                # Subtract mean and divide by standard deviation for each feature
                dataset[i] = (dataset[i] - np.mean(dataset[i], axis=0)) / np.std(dataset[i], axis=0)

        # Initialize an array to store output likelihoods for each observation
        output = np.zeros(len(dataset))

        # Iterate over each observation in the dataset
        for i, observation in enumerate(dataset):
            # Calculate the probability of the observation given the model
            obs_prob = self.calculate_observation_probability(observation)
            # Perform forward pass to calculate alpha values
            alpha = self.forward_pass(obs_prob)
            # Compute the likelihood of the observation using alpha values
            # The likelihood is the sum of exponentials of alpha values for each state
            output[i] = np.sum(np.exp(alpha), axis=0)[-1]

        # Return the array of likelihoods for all observations
        return output



def build_dataset():
    files = sorted(os.listdir(sound_path))
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    data = dict()
    n = len(files)
    for i in range(n):
        feature = feature_extractor(sound_path=sound_path + files[i])
        digit = files[i][0]
        if digit not in data.keys():
            data[digit] = []
            x_test.append(feature)
            y_test.append(digit)
        else:
            if np.random.rand() < 0.25:
                x_test.append(feature)
                y_test.append(digit)
            else:
                x_train.append(feature)
                y_train.append(digit)
                data[digit].append(feature)
    return x_train, y_train, x_test, y_test, data

def feature_extractor(sound_path):
    sampling_freq, audio = wavfile.read(sound_path)
    mfcc_features = mfcc(audio, sampling_freq,nfft = 2048,numcep=13,nfilt=13)
    return mfcc_features

def plot_confusion_matrix(test_labels, classifier_labels, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.figure()

    cm = confusion_matrix(test_labels, classifier_labels)

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # title += '\nCCR is = ' + str(CCR(test_labels, classifier_labels))

    print(title)

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


x_train, y_train, x_test, y_test, data = build_dataset()

nstates = []
fscore = []



predictedLables = []
models = {}
predictions ={}

for key in data.keys():
    model = GMM_HMM(key,number_of_states,number_of_gaussians)
    model.train(data[key],1)
    models[key] = model
    predictions[key] = model.likelihood(x_test)

correct = 0
total = 0

TruePositives = 0
FalsePositives = 0
FalseNegatives = 0



for i in range(len(x_test)):
            predict = -int(1e9)
            predict_digit = -1
            for key in data.keys():
                if predictions[key][i] > predict:
                    predict = predictions[key][i]
                    predict_digit = key
            if predict_digit == y_test[i]:
                TruePositives += 1
                correct += 1
            else:
                FalsePositives += 1
                FalseNegatives += 1
            total += 1
            predictedLables.append(predict_digit)

plot_confusion_matrix(y_test,predictedLables,data.keys())
print("Accuray :", correct/total)
print("Precision: ", TruePositives/(TruePositives + FalsePositives))
print("Recall: ", TruePositives/(TruePositives + FalseNegatives))
print("F1 Score: ", 2*TruePositives/(2*TruePositives + FalsePositives + FalseNegatives))



