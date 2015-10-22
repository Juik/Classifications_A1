import numpy as np
from check_grad import check_grad
from plot_digits import *
from utils import *
from logistic import *

#TO PLOT THE CE/FR OF EACH ITER FOR CHECKING
def plotce(it, ce_train, ce_valid):
    plt.figure(1)
    plt.plot(it, ce_train, marker='o', label='Training Set')
    plt.plot(it, ce_valid, marker='x', label='Validation Set')
    legend = plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('Cross Entropy')
    plt.grid()
    plt.show()

def plotcr(it, fr_train, fr_valid):
    plt.figure(2)
    plt.plot(it, fr_train, marker='o', label='Training Set')
    plt.plot(it, fr_valid, marker='x', label='Validation Set')
    legend = plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('Classfication Rate')
    plt.grid()
    plt.show()

#TO SEE HOW WEIGHTS DISTRIBUTES
def plotweight(weights):
    dict = {}
    for weight in np.nditer(weights):
        weight = str(round(weight,3))
        if(dict.has_key(weight)==0):
            dict2 = {weight: 1}
            dict.update(dict2)
        else:
            dict[weight] = dict.get(weight)+1
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


def run_logistic_regression_pen():
    
    train_inputs_s, train_targets_s = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()
    train_inputs, train_targets = load_train()

    N, M = train_inputs.shape

    l_train_cf_err = []
    l_train_cf_err_s = []
    l_train_co_en = []
    l_train_co_en_s =[]
    l_valid_cf_err = []
    l_valid_cf_err_s = []
    l_valid_co_en = []
    l_valid_co_en_s=[]

    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': 0.1,
                    'iterations': 1000,                    
                    #firstly set 1, we change it later
                    'weight_regularization': 1,
                    'iterations_s': 1000
                 }   
    rerun = 10
    #Actually it's lambda but due to sensitive word...
    l_s = [0.0001,0.001,0.01,0.1,1.0,10.0,100.0]


    for l in l_s:
        print 'this round lambda equals: %.4f' % l
        #calculate for 10 times and average each
        l_train_cf_err_tmp = 0
        l_train_cf_err_s_tmp = 0
        l_train_co_en_tmp = 0
        l_train_co_en_s_tmp = 0
        l_valid_cf_err_tmp = 0
        l_valid_cf_err_s_tmp = 0
        l_valid_co_en_tmp = 0
        l_valid_co_en_s_tmp = 0

        for run in range(rerun):
            # Logistic regression weights
            # TODO:Initialize to random weights here.
            weights = 0.1*np.random.randn(M+1,1)
            weights_s = 0.1*np.random.randn(M+1,1)

            '''
            IF YOU WANT TO SEE THE INITIAL WEIGHTS DISTRIBUTION
            plotweight(weights)
            '''

            # Verify that your logistic function produces the right gradient.
            # diff should be very close to 0.
            run_check_grad(hyperparameters)

            last_call = hyperparameters['iterations_s']-1 
            last_cal  = hyperparameters['iterations']-1
            hyperparameters['weight_regularization'] = l

            ce_train = np.zeros(last_cal+1)
            ce_valid = np.zeros(last_cal+1)
            fr_train = np.zeros(last_cal+1)
            fr_valid = np.zeros(last_cal+1)
            ce_train_s = np.zeros(last_call+1)
            ce_valid_s = np.zeros(last_call+1)
            fr_train_s = np.zeros(last_call+1)
            fr_valid_s = np.zeros(last_call+1)

            it = np.zeros(last_cal+1)
            it_s = np.zeros(last_call+1)
            
            # Begin learning with gradient descent
            # first let's do small set
            for t in xrange(hyperparameters['iterations_s']):

                it_s[t] = t

                # TODO: you may need to modify this loop to create plots, etc.

                # Find the negative log likelihood and its derivatives w.r.t. the weights.
                f_s, df_s, predictions_s = logistic_pen(weights_s, train_inputs_s, train_targets_s, hyperparameters)

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
            
                # print some stats
                
                if t == hyperparameters['iterations']-1:
                    stat_msg = "ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f}  "
                    stat_msg += "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}"
                    print stat_msg.format(t+1,
                                          float(f_s / N),
                                          float(cross_entropy_train_s),
                                          float(frac_correct_train_s*100),
                                          float(cross_entropy_valid_s),
                                          float(frac_correct_valid_s*100))
                

            '''
            DEANNOTATE IF YOU WANT TO SEE THE PICTURE OF EACH ROUND
            if(t==last_cal):
                plotce(it_s, ce_train_s, ce_valid_s)
                plotcr(it_s, fr_train_s, fr_valid_s)
            '''

            #print '---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------now start with the bigger dataset------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
            
            # Begin learning with gradient descent
            # Now let's Rock n Roll a bigger dataset
            for t in xrange(hyperparameters['iterations']):

                it[t] = t

                # TODO: you may need to modify this loop to create plots, etc.

                # Find the negative log likelihood and its derivatives w.r.t. the weights.
                f, df, predictions = logistic_pen(weights, train_inputs, train_targets, hyperparameters)
                
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

                # print some stats
                
                if t == hyperparameters['iterations']-1:
                    stat_msg = "ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f}  "
                    stat_msg += "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}"
                    print stat_msg.format(t+1,
                                          float(f / N),
                                          float(cross_entropy_train),
                                          float(frac_correct_train*100),
                                          float(cross_entropy_valid),
                                          float(frac_correct_valid*100))
                

            '''
            DEANNOTATE IF YOU WANT TO SEE THE PICTURE OF EACH ROUND
            if(t==last_call):
                plotce(it_s, ce_train_s, ce_valid_s)
                plotcr(it_s, fr_train_s, fr_valid_s)
            '''

            #add up all the result every round after training and validating
            l_valid_co_en_tmp += ce_valid[last_call]
            l_valid_co_en_s_tmp += ce_valid_s[last_call]
            l_valid_cf_err_tmp += fr_valid[last_cal]
            l_valid_cf_err_s_tmp += fr_valid_s[last_call]
            l_train_co_en_tmp += ce_train[last_cal]
            l_train_co_en_s_tmp += ce_train_s[last_call]
            l_train_cf_err_tmp += fr_train[last_cal]
            l_train_cf_err_s_tmp += fr_train_s[last_call]

        #let's start trying test dataset
        pre_test = logistic_predict(weights, test_inputs)
        pre_test_s = logistic_predict(weights_s, test_inputs)
        ce_test, fc_test = evaluate(test_targets, pre_test)
        ce_test_s, fc_test_s = evaluate(test_targets, pre_test_s)

        # print some stats
        stat_msg = " TEST CE:{:.6f}  "
        stat_msg += "TEST FRAC:{:2.2f}  TEST_S CE:{:.6f}  TEST_S FRAC:{:2.2f}"
        print stat_msg.format(
                              float(ce_test),
                              float(fc_test*100),
                              float(ce_test_s),
                              float(fc_test_s*100))

        '''
        IF YOU WANT TO SEE THE INITIAL WEIGHTS DISTRIBUTION
        
        plotweight(weights)
        '''

        #get the avg of trained result and return to list of cross-entrophy or
        #classfication rate so that we can plot as a picture
        avg_train_ce = l_train_co_en_tmp / (rerun * len(l_s))
        avg_train_ce_s = l_train_co_en_s_tmp /  (rerun * len(l_s))
        avg_train_cf = l_train_cf_err_tmp / rerun
        avg_train_cf_s = l_train_cf_err_s_tmp /rerun
        avg_valid_ce = l_valid_co_en_tmp / (rerun * len(l_s))
        avg_valid_ce_s = l_valid_co_en_s_tmp / (rerun * len(l_s))
        avg_valid_cf = l_valid_cf_err_tmp / rerun
        avg_valid_cf_s = l_valid_cf_err_s_tmp / rerun

        #then we put them in a list and let's plot them as "l" changes in a pic
        l_train_cf_err.append(avg_train_cf)
        l_train_cf_err_s.append(avg_train_cf_s)
        l_train_co_en.append(avg_train_ce)
        l_train_co_en_s.append(avg_train_ce_s)
        l_valid_cf_err.append(avg_valid_cf)
        l_valid_cf_err_s.append(avg_valid_cf_s)
        l_valid_co_en.append(avg_valid_ce)
        l_valid_co_en_s.append(avg_valid_ce_s)


    s = 'learning rate is %.2f and iter is %d' % (hyperparameters['learning_rate'], hyperparameters['iterations'])
    #plot normal mnist_train set
    plt.figure(1)
    plt.title('Cross Entropy for normal Dataset + %s' % s)
    plt.plot(l_s, l_train_co_en, marker='.', label='Training Set')
    plt.plot(l_s, l_valid_co_en, marker='D', label='Validation Set')
    legend = plt.legend()
    plt.xscale('log')
    plt.yscale('linear')
    plt.xlabel('lambda')
    plt.ylabel('Cross Entropy')
    plt.grid()

    plt.figure(2)
    plt.title('Classfication Rate for normal Dataset + %s' % s)
    plt.plot(l_s, l_train_cf_err, 'r-', label='Training Set')
    plt.plot(l_s, l_valid_cf_err, marker='x', label='Validation Set')
    legend = plt.legend()
    plt.xscale('log')
    plt.yscale('linear')
    plt.xlabel('lambda')
    plt.ylabel('Classification Rate')
    plt.axis([0.001,110,0.45,1.05])
    plt.grid()


    #plot small mnist_train set
    plt.figure(3)
    plt.title('Cross Entropy for small Dataset + %s'  % s)
    plt.plot(l_s, l_train_co_en_s, marker='.', label='Training Set')
    plt.plot(l_s, l_valid_co_en_s, marker='D', label='Validation Set')
    legend = plt.legend()
    plt.xscale('log')
    plt.yscale('linear')
    plt.xlabel('lambda')
    plt.ylabel('Cross Entropy')
    plt.grid()

    plt.figure(4)
    plt.title('Classfication Rate for small Dataset + %s'  % s)
    plt.plot(l_s, l_train_cf_err_s, 'r-', label='Training Set')
    plt.plot(l_s, l_valid_cf_err_s, marker='x', label='Validation Set')
    legend = plt.legend()
    plt.xscale('log')
    plt.yscale('linear')
    plt.xlabel('lambda')
    plt.ylabel('Classification Rate')
    plt.axis([0.001,110,0.45,1.05])
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
    run_logistic_regression_pen()
    raw_input("Press Enter to continue...")