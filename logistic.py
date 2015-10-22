""" Methods for doing logistic regression."""

import numpy as np
import matplotlib.pyplot as plt
from check_grad import check_grad
from utils import *
from logistic import *

def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities. This is the output of the classifier.
    """

    # TODO: Finish this function
    (n,m) = data.shape
    x0 = np.ones((n,1))
    #we need to add one colunm 1 to make data and weights same num of col
    cmp_data = np.concatenate((data,x0),axis=1)
    # y = WTX
    wx = cmp_data.dot(weights)
    z = sigmoid(wx)
    return z


#this func helps do evaluate
def compare_two_lists(aa,bb):
    count = 0
    correct_count = 0
    for a in aa:
        if a== bb[count]:
            correct_count = correct_count + 1
        count = count + 1

    return correct_count / float(count)

def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of binary targets. Values should be either 0 or 1
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy.  CE(p, q) = E_p[-log q].  Here
                       we want to compute CE(targets, y).
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    # TODO: Finish this function
    '''
    print targets.T 
    print targets
    '''
    #found from wiki
    ce = -np.dot(targets.T, np.log(y))-np.dot(1-targets.T, np.log(1-y))
    frac_correct = compare_two_lists(np.around(y),targets)

    return ce, frac_correct

def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of binary targets. Values should be either 0 or 1.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of derivative of f w.r.t. weights.
        y:       N x 1 vector of probabilities.
    """

    # TODO: Finish this function
    (n,m)=data.shape
    x0 = np.ones((n,1))
    cmp_data = np.concatenate((data,x0),axis=1)
    y = logistic_predict(weights,data)

    #f is the loss function in the slides
    f =  -np.dot(targets.T, np.log(y))-np.dot(1-targets.T, np.log(1-y))
    #df is the final simplying after gradient and differentiation
    df = np.dot(cmp_data.T,(1-targets)-(1-y))
    
    # print 'f'
    # print f
    # print 'df'
    # print df
    # print 'y'
    # print y

    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of binary targets. Values should be either 0 or 1.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of derivative of f w.r.t. weights.
    """

    # TODO: Finish this function
    (n,m)=data.shape
    x0 = np.ones((n,1))
    cmp_data = np.concatenate((data,x0),axis=1)
    y = logistic_predict(weights,data)
    alpha = hyperparameters['weight_regularization']
    
    #f is the loss function in the slides plus lambda/2 * sigma (wi^2)
    f =  -np.dot(targets.T, np.log(y))-np.dot(1-targets.T, np.log(1-y)) +0.5*alpha*(np.dot(weights.T,weights))
    #df is the final simplying after gradient and differentiation plus lambda wi
    df = np.dot(cmp_data.T,(1-targets)-(1-y)) + alpha * weights


    return f, df, y

if __name__ == '__main__':
    '''
    ATTENTION: This main function is just for tesing 
    We will never use functions below. DO NOT RUN THIS FILE! IGNORE HERE!
    '''

    '''
    train_inputs, train_targets = load_train()
    train_inputs_s, train_targets_s = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    N, M = train_inputs.shape

    #Gonna try weight_regularization for {0.001,0.01,0.1,1}
    hyperparameters = {'learning_rate': 0.10,
                    'weight_regularization': 1,
                    'num_iterations': 1000,
                    'num_iterations_s': 1000
                 }

    weights = np.random.random_sample((M+1,1))
    weights_s = np.random.random_sample((M+1,1))
    print'--------------'
    print weights
    print'--------------'
    print weights_s
    print'--------------'
    ce_train = np.zeros(hyperparameters['num_iterations'])
    ce_valid = np.zeros(hyperparameters['num_iterations'])
    ce_train_s = np.zeros(hyperparameters['num_iterations_s'])
    ce_valid_s = np.zeros(hyperparameters['num_iterations_s'])
    it = np.zeros(hyperparameters['num_iterations'])
    it_s = np.zeros(hyperparameters['num_iterations_s'])

    print('ready to start')
    y = logistic_predict(weights,train_inputs)
    ce, frac_correct = evaluate(train_targets,y)
    print('ce is: ')
    print ce
    print('frac_correct is:')
    print frac_correct
    logistic(weights,train_inputs_s,train_targets_s,hyperparameters)
    logistic_pen(weights,train_inputs_s,train_targets_s,hyperparameters)
    '''