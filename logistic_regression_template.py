import numpy as np
from check_grad import check_grad
from plot_digits import *
from utils import *
from logistic import *


def plotweight(weights):
    dict = {}
    for weight in np.nditer(weights):
        weight = str(round(weight,3))
        print 'this round weight is %s' % weight
        if(dict.has_key(weight)==0):
            dict2 = {weight: 1}
            dict.update(dict2)
            print dict
        else:
            dict[weight] = dict.get(weight)+1
            # dict2 = {weight : dict.get(weight)+1}
            # dict.update(dict)
            print dict
    print dict
    random_weight = []
    freq = []

    for key,value in dict.items():
        random_weight.append(float(key))
        freq.append(value)

    rects1 = plt.bar(random_weight, freq, width=0.0008 ,color='b')  
    plt.xlabel('weight')  
    plt.ylabel('number of frequence')  
    plt.title('distribution of weight frequence')  
    # plt.ylim(0,40)  
    plt.show()  



def run_logistic_regression():
    train_inputs, train_targets = load_train()
    train_inputs_s, train_targets_s = load_train_small()
    #train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    N, M = train_inputs.shape

    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': 0.1,
                    'iterations': 1000,
                    'iterations_s': 1000
                 }

    # Logistic regression weights
    # TODO:Initialize to random weights here.
    weights = 0.1*np.random.randn(M+1,1)
    weights_s = 0.1* np.random.randn(M+1,1)
    # weights = 0.1*np.random.random_sample((M+1,1))
    # weights_s = 0.1*np.random.random_sample((M+1,1))

    # plotweight(weights)
    # plotweight(weights_s)

    ce_train = np.zeros(hyperparameters['iterations'])
    ce_valid = np.zeros(hyperparameters['iterations'])
    ce_train_s = np.zeros(hyperparameters['iterations_s'])
    ce_valid_s = np.zeros(hyperparameters['iterations_s'])
    fr_train_s = np.zeros(hyperparameters['iterations_s'])
    fr_valid_s = np.zeros(hyperparameters['iterations_s'])
    fr_train = np.zeros(hyperparameters['iterations_s'])
    fr_valid = np.zeros(hyperparameters['iterations_s'])
    it = np.zeros(hyperparameters['iterations'])
    it_s = np.zeros(hyperparameters['iterations_s'])

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    # first let's do small set
    for t in xrange(hyperparameters['iterations_s']):

        # TODO: you may need to modify this loop to create plots, etc.

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f_s, df_s, predictions_s = logistic(weights_s, train_inputs_s, train_targets_s, hyperparameters)

        # Evaluate the prediction.
        cross_entropy_train_s, frac_correct_train_s = evaluate(train_targets_s, predictions_s)

        if np.isnan(f_s) or np.isinf(f_s):
            raise ValueError("nan/inf error")

        # update parameters
        weights_s = weights_s - hyperparameters['learning_rate'] * df_s / N

        # Make a prediction on the valid_inputs.
        predictions_valid_s = logistic_predict(weights_s, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid_s, frac_correct_valid_s = evaluate(valid_targets, predictions_valid_s)

        ce_train_s[t] = cross_entropy_train_s
        ce_valid_s[t] = cross_entropy_valid_s
        fr_train_s[t] = frac_correct_train_s
        fr_valid_s[t] = frac_correct_valid_s
        it_s[t] = t

        # print some stats only for the last round
        if t == hyperparameters['iterations']-1:
            stat_msg = "ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f}  "
            stat_msg += "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}"
            print stat_msg.format(t+1,
                                  float(f_s / N),
                                  float(cross_entropy_train_s),
                                  float(frac_correct_train_s*100),
                                  float(cross_entropy_valid_s),
                                  float(frac_correct_valid_s*100))

    print '---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------now start with the bigger dataset------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
    
    # Begin learning with gradient descent
    # Now let's Rock n Roll a bigger dataset
    for t in xrange(hyperparameters['iterations']):

        # TODO: you may need to modify this loop to create plots, etc.

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic(weights, train_inputs, train_targets, hyperparameters)
        
        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        weights = weights - hyperparameters['learning_rate'] * df / N

        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)
        
        ce_train[t] = cross_entropy_train
        ce_valid[t] = cross_entropy_valid
        fr_train[t] = frac_correct_train
        fr_valid[t] = frac_correct_valid

        it[t] = t
        
        # print some stats only for the last round
        if t == hyperparameters['iterations']-1:
            stat_msg = "ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f}  "
            stat_msg += "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}"
            print stat_msg.format(t+1,
                                  float(f / N),
                                  float(cross_entropy_train),
                                  float(frac_correct_train*100),
                                  float(cross_entropy_valid),
                                  float(frac_correct_valid*100))
        
    # plotweight(weights)

    #let's start trying test dataset
    pre_test = logistic_predict(weights, test_inputs)
    pre_test_s = logistic_predict(weights_s, test_inputs)
    ce_test, fc_test = evaluate(test_targets, pre_test)
    ce_test_s, fc_test_s = evaluate(test_targets, pre_test_s)    
    
    
    # print some test stats
 
    stat_msg = " test CE:{:.6f}  "
    stat_msg += "test FRAC:{:2.2f} test Error:{:2.2f} test_s CE:{:.6f}  test_s FRAC:{:2.2f} test_s Error:{:2.2f}"
    print stat_msg.format(
                          float(ce_test),
                          float(fc_test*100),
                          float((1-fc_test)*100),
                          float(ce_test_s),
                          float(fc_test_s*100),
                          float((1-fc_test_s)*100))
    
    #after training see the distribution of weights
    # plotweight(weights) 

    #Below is we can see what's going on for this hyperparameter
    plt.figure(figsize=(8,7),dpi=98)
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)

    ax1.set_title('mnist_train')
    ax1.plot(it, ce_train, marker='o', label='Training Set')
    ax1.plot(it, ce_valid, marker='x', label='Validation Set')
    legend = plt.legend()
    ax1.set_ylabel('Cross Entropy')
    ax1.legend(loc='upper right')
    ax1.grid()


    ax2.set_title('mnist_train_small')
    ax2.plot(it_s, ce_train_s, marker='o', label='Training Set')
    ax2.plot(it_s, ce_valid_s, marker='x', label='Validation Set')
    legend = plt.legend()
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Cross Entropy')
    ax2.grid()

    ax3.set_title('mnist_train')
    ax3.plot(it, fr_train, marker='o', label='Training Set')
    ax3.plot(it, fr_valid, marker='x', label='Validation Set')
    ax3.set_ylabel('correction rate')
    ax3.grid()

    ax4.set_title('mnist_train_small')
    ax4.plot(it, fr_train_s, marker='o', label='Training Set')
    ax4.plot(it, fr_valid_s, marker='x', label='Validation Set')
    ax4.set_ylabel('correction rate')
    ax4.grid()

    plt.show()


    #Below two images to answer question 2.2. Plot cross entropy
    plt.figure(1)
    plt.title('mnist_train')
    plt.plot(it, ce_train, 'r-', label='Training Set')
    plt.plot(it, ce_valid, 'x', label='Validation Set')
    legend = plt.legend()
    plt.xlabel('iter')
    plt.ylabel('Cross Entropy')
    plt.grid()

    plt.figure(2)
    plt.title('mnist_train_small')
    plt.plot(it, ce_train_s, 'r-', label='Training Set')
    plt.plot(it, ce_valid_s, 'x', label='Validation Set')
    legend = plt.legend()
    plt.xlabel('iter')
    plt.ylabel('Cross Entropy')
    plt.grid()

    plt.show()

    


def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    """

    # This creates small random data with 20 examples and 
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions+1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.round(np.random.rand(num_examples, 1), 0)

    diff = check_grad(logistic,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyperparameters)

    print "diff =", diff

if __name__ == '__main__':
    run_logistic_regression()
    raw_input("Press Enter to continue...")
    '''
    Below for testing plot weight function
    
    train_inputs, train_targets = load_train()
    train_inputs_s, train_targets_s = load_train_small()
    #train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    N, M = train_inputs.shape
    # weights = 0.1*np.random.random_sample(M+1)
    weights = 0.1*np.random.randn(M+1)
    plotweight(weights)
    '''