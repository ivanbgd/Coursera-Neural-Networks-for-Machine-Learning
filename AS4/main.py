import numpy as np
from collections import namedtuple, OrderedDict
from utils import *

report_calls_to_sample_bernoulli = None
data_sets = None
temp = None


def sample_bernoulli(probabilities):
    """
    Returns a "binary" matrix of the same shape as 'probabilities'.
    It is binary in the sense that it contains only 0s and 1s.
    This creates a Bernoulli distribution over samples from the file 'a4_randomness_source.mat'.
    """
    if report_calls_to_sample_bernoulli:
        print('sample_bernoulli() was called with a matrix of size {} by {}.'.format(probabilities.shape[0], probabilities.shape[1]))
    seed = np.sum(probabilities)
    binary = 1 * (probabilities > a4_rand(probabilities.shape, seed))   # The "1*" is to avoid the "logical" data type, which just confuses things.
    return binary


def a4_main(n_hid, lr_rbm, lr_classification, n_iterations):
    # first, train the rbm
    global report_calls_to_sample_bernoulli
    report_calls_to_sample_bernoulli = False

    rbm_w = optimize([n_hid, 256],
                     lambda rbm_w, data: cd1(rbm_w, data.inputs),  # discard labels
                     data_sets.training,
                     lr_rbm,
                     n_iterations)
    # rbm_w is now a weight matrix of <n_hid> by <number of visible units, i.e. 256>
    show_rbm(rbm_w)

    input_to_hid = rbm_w
    # calculate the hidden layer representation of the labeled data
    hidden_representation = logistic(np.dot(input_to_hid, data_sets.training.inputs))

    # train hid_to_class
    data_2_inputs = hidden_representation
    data_2_targets = data_sets.training.targets
    Data_2 = namedtuple('Data_2', ['inputs', 'targets'])
    data_2 = Data_2(data_2_inputs, data_2_targets)
    hid_to_class = optimize([10, n_hid], lambda model, data: classification_phi_gradient(model, data), data_2, lr_classification, n_iterations)

    # report results
    data_details = OrderedDict()
    data_details['training'] = data_sets.training
    data_details['validation'] = data_sets.validation
    data_details['test'] = data_sets.test
    for item in data_details.items():
        data_name = item[0]
        data = item[1]
        hid_input = np.dot(input_to_hid, data.inputs)   # size: <number of hidden units> by <number of data cases>
        hid_output = logistic(hid_input)                # size: <number of hidden units> by <number of data cases>
        class_input = np.dot(hid_to_class, hid_output)  # size: <number of classes> by <number of data cases>
        class_normalizer = log_sum_exp_over_rows(class_input) # log(sum(exp of class_input)) is what we subtract to get properly normalized log class probabilities. size: <1> by <number of data cases>
        log_class_prob = class_input - np.tile(class_normalizer, (class_input.shape[0], 1))     # log of probability of each class. size: <number of classes, i.e. 10> by <number of data cases>
        error_rate = np.mean(argmax_over_rows(class_input) != argmax_over_rows(data.targets))   # scalar
        loss = -np.mean(np.sum(log_class_prob * data.targets, axis=0))   # scalar. select the right log class probability using that sum then take the mean over all data cases.
        print('For the {} data, the classification cross-entropy loss is {}, and the classification error rate (i.e. the misclassification rate) is {}.'.format(data_name, loss, error_rate))
    print()
        
    report_calls_to_sample_bernoulli = True
    return



report_calls_to_sample_bernoulli = False

training, validation, test = load_data()    # same as in PA3

Datas = namedtuple('Datas', ['training', 'validation', 'test'])
data_sets = Datas(training, validation, test)

test_rbm_w = a4_rand([100, 256], 0) * 2 - 1
small_test_rbm_w = a4_rand([10, 256], 0) * 2 - 1

temp = extract_mini_batch(data_sets.training, 1-1, 1)
data_1_case = sample_bernoulli(temp.inputs)
temp = extract_mini_batch(data_sets.training, 100-1, 10)
data_10_cases = sample_bernoulli(temp.inputs)
temp = extract_mini_batch(data_sets.training, 200-1, 37)
data_37_cases = sample_bernoulli(temp.inputs)

test_hidden_state_1_case = sample_bernoulli(a4_rand([100, 1], 0))
test_hidden_state_10_cases = sample_bernoulli(a4_rand([100, 10], 1))
test_hidden_state_37_cases = sample_bernoulli(a4_rand([100, 37], 2))

report_calls_to_sample_bernoulli = True

del temp


