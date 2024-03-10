import math
from scipy.stats import multivariate_normal
import numpy as np


class GMM:
    def __init__(self,X,k,mu,cov,pi_s,num_iters,tol,cov_type='full'):
        '''
        X : Datapoints of dimension (n,d)
        k: Number of clusters
        mu: Intial means of dimension (k,d)
        cov: Initial covariance matrix of shape 
             (k,d,d) if full
             (k,) if spherical [sigma_{k}^2I]
             (d,d) if tied [Same covariance matrices for all]
             (k,d) if its diagonal [variance for each feature along diagonal]
        pi_s: Intial weights for gaussian mixtures
        num_iters: Maximum number of iterations for which the algorithm should be run
        cov_type: Type of covariance matrix 
                  full - each mixture can have arbitraty covariance matrix
                  tied - each mixture has same covariance matrix
                  diagonal - each mixture has a diagonal covariance matrix with separate variance for each component
                  spherical - each mixture k has a covariance matrix of type [sigma_{k}^2I]
        tol: tolerance for less than which its considered converged
        '''

        self.X = np.array(X)
        self.mu = mu
        self.covariances = cov
        self.pis = pi_s
        self.iters = num_iters
        self.covariance_type = cov_type
        self.num_components = k
        self.n , self.d = self.X.shape
        self.wik = np.zeros((self.n,self.num_components))
        self.log_likelihood_history = []

    def E_Step(self):
        '''
        E_Step for GMM, calculates the responsibilities, w_{i}_{k}
        returns log_likelihood
        '''
        log_likelihood = 0.0
        for i in range(self.n):
            tot = 0.0
            # Calculate the denominor for responsibilities for each X{i}
            for k in range(self.num_components):
                tot = tot + self.pis[k]*multivariate_normal.pdf(self.X[i],self.mu[k],self.covariances[k])
            log_likelihood += math.log(tot)
            for k in range(self.num_components):
                single = self.pis[k]*multivariate_normal.pdf(self.X[i],self.mu[k],self.covariances[k])
                self.wik[i,k] = single/tot
        return log_likelihood
    
    def M_Step(self):
        '''
        M_Step for GMM, finds the updated Means, covariance and weights of components
        '''
        sum_responsibilities = np.sum(self.wik,axis=0)

        # Finding the means for kth component
        for k in range(self.num_components):
            resp_k = self.wik[:,k].reshape(-1,1)
            numerator = np.sum(resp_k * self.X,axis=0)
            self.mu[k] = numerator/sum_responsibilities[k]

        # For covariances - could be written as a single vector expression, but that would be less readable
        for k in range(self.num_components):
            self.covariances[k] = np.zeros((self.d, self.d))
            for i in range(self.n):
                self.covariances[k] += self.wik[i, k] * (np.vstack(self.X[i] - self.mu[k]) * (self.X[i]-self.mu[k]))
            self.covariances[k] /= sum_responsibilities[k]


    def fit(self):
        '''
        Does EM algorithm steps till it converges or till iterations get over
        '''
        log_likelihood = 0.0
        for iter in range(self.iters):
            log_likelihood_1 = self.E_Step()
            self.M_Step()
            if(abs(log_likelihood-log_likelihood_1)<=self.tol):
                print("GMM converged at iteration",iter)
                break
            log_likelihood = log_likelihood_1
            self.log_likelihood_history.append(log_likelihood)

    def get_score(self,X):
        '''
        X : Datapoints for which log likelihood score has to be calculated
        '''
        log_likelihood = 0.0
        for i in range(self.n):
            tot = 0.0
            # Calculate the denominor for responsibilities for each X{i}
            for k in range(self.num_components):
                tot = tot + self.pis[k]*multivariate_normal.pdf(self.X[i],self.mu[k],self.covariances[k])
            log_likelihood += math.log(tot)
        return log_likelihood