import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from plot_perceptron import *


def learn_perceptron(neg_examples_nobias, pos_examples_nobias, w_init, w_gen_feas):
    """ Learns the weights of a perceptron and displays the results.
        Learns the weights of a perceptron for a 2-dimensional dataset and plots
        the perceptron at each iteration where an iteration is defined as one
        full pass through the data. If a generously feasible weight vector
        is provided then the visualization will also show the distance
        of the learned weight vectors to the generously feasible weight vector.
        Required Inputs:
          neg_examples_nobias - The num_neg_examples x 2 matrix for the examples with target 0.
              num_neg_examples is the number of examples for the negative class.
          pos_examples_nobias - The num_pos_examples x 2 matrix for the examples with target 1.
              num_pos_examples is the number of examples for the positive class.
          w_init - A 3-dimensional initial weight vector. The last element is the bias.
          w_gen_feas - A generously feasible weight vector.
        Returns:
          w - The learned weight vector.
    """
    # Bookkeeping
    num_neg_examples = neg_examples_nobias.shape[0]
    num_pos_examples = pos_examples_nobias.shape[0]
    num_err_history = np.array([], dtype=np.int)    # doesn't have to be np.array, can be a Python list
    w_dist_history = np.array([])                   # doesn't have to be np.array, can be a Python list

    # Here we add a column of ones to the examples in order to allow us to learn bias parameters.
    neg_examples = np.concatenate((neg_examples_nobias, np.ones((num_neg_examples, 1))), axis=1)
    pos_examples = np.concatenate((pos_examples_nobias, np.ones((num_pos_examples, 1))), axis=1)

    # If weight vectors have not been provided, initialize them appropriately.
    if (not('w_init' in locals()) or (w_init.size == 0)):
        w = np.random.randn(3, 1)
    else:
        w = np.array(w_init)

    if (not('w_gen_feas' in locals())):
        w_gen_feas = np.array([])

    # Find the data points that the perceptron has incorrectly classified and record the number of errors it makes.
    iter = 0
    mistakes0, mistakes1 = eval_perceptron(neg_examples, pos_examples, w)
    num_errs = mistakes0.shape[0] + mistakes1.shape[0]
    num_err_history = np.append(num_err_history, num_errs)  # copies the array to a new one
    print("Number of errors in iteration {}:\t{}".format(iter, num_errs))
    print("Weights:\n{}".format(w))
    plot_perceptron(neg_examples, pos_examples, mistakes0, mistakes1, num_err_history, w, w_dist_history)
    key = input('<Press enter to continue, q to quit.>')
    if (key == 'q'):
        return w

    # If a generously feasible weight vector exists, record the distance to it from the initial weight vector.
    if (w_gen_feas.size > 0):
        w_dist_history = np.append(w_dist_history, np.linalg.norm(w - w_gen_feas))  # copies the array to a new one

    # Iterate until the perceptron has correctly classified all points.
    while (num_errs > 0):
        iter += 1

        # Update the weights of the perceptron.
        w = update_weights(neg_examples, pos_examples, w)

        # If a generously feasible weight vector exists, record the distance to it from the initial weight vector.
        if (w_gen_feas.size > 0):
            w_dist_history = np.append(w_dist_history, np.linalg.norm(w - w_gen_feas))

        # Find the data points that the perceptron has incorrectly classified and record the number of errors it makes.
        mistakes0, mistakes1 = eval_perceptron(neg_examples, pos_examples, w)
        num_errs = mistakes0.shape[0] + mistakes1.shape[0]
        num_err_history = np.append(num_err_history, num_errs)

        print("Number of errors in iteration {}:\t{}".format(iter, num_errs))
        print("Weights:\n{}".format(w))
        plot_perceptron(neg_examples, pos_examples, mistakes0, mistakes1, num_err_history, w, w_dist_history)
        key = input('<Press enter to continue, q to quit.>')
        if (key == 'q'):
            break

    return w


# WRITE THE CODE TO COMPLETE THIS FUNCTION
def update_weights(neg_examples, pos_examples, w_current):
    """ Updates the weights of the perceptron for incorrectly classified points
        using the perceptron update algorithm. This function makes one sweep
        over the dataset.
        Inputs:
          neg_examples - The num_neg_examples x 3 matrix for the examples with target 0.
              num_neg_examples is the number of examples for the negative class.
          pos_examples- The num_pos_examples x 3 matrix for the examples with target 1.
              num_pos_examples is the number of examples for the positive class.
          w_current - A 3-dimensional weight vector, the last element is the bias.
        Returns:
          w - The weight vector after one pass through the dataset using the perceptron
              learning rule.
    """
    w = np.copy(w_current)
    num_neg_examples = neg_examples.shape[0]
    num_pos_examples = pos_examples.shape[0]

    for i in range(num_neg_examples):
        this_case = neg_examples[i, :].reshape(1, -1)
        x = this_case.T
        activation = np.dot(this_case, w)
        if (activation >= 0):
            # YOUR CODE HERE

    for i in range(num_pos_examples):
        this_case = pos_examples[i, :].reshape(1, -1)
        x = this_case.T
        activation = np.dot(this_case, w)
        if (activation < 0):
            # YOUR CODE HERE

    return w


def eval_perceptron(neg_examples, pos_examples, w):
    """ Evaluates the perceptron using a given weight vector. Here, evaluation
        refers to finding the data points that the perceptron incorrectly classifies.
        Inputs:
          neg_examples - The num_neg_examples x 3 matrix for the examples with target 0.
              num_neg_examples is the number of examples for the negative class.
          pos_examples- The num_pos_examples x 3 matrix for the examples with target 1.
              num_pos_examples is the number of examples for the positive class.
          w - A 3-dimensional weight vector, the last element is the bias.
        Returns:
          mistakes0 - A vector containing the indices of the negative examples that have been
              incorrectly classified as positive.
          mistakes0 - A vector containing the indices of the positive examples that have been
              incorrectly classified as negative.
    """
    num_neg_examples = neg_examples.shape[0]
    num_pos_examples = pos_examples.shape[0]
    mistakes0 = np.array([], dtype=np.int)  # doesn't have to be np.array, can be a Python list
    mistakes1 = np.array([], dtype=np.int)  # doesn't have to be np.array, can be a Python list

    for i in range(num_neg_examples):
        this_case = neg_examples[i, :]
        activation = np.dot(this_case, w)
        if (activation >= 0):
            mistakes0 = np.append(mistakes0, i)

    for i in range(num_pos_examples):
        this_case = pos_examples[i, :]
        activation = np.dot(this_case, w)
        if (activation < 0):
            mistakes1 = np.append(mistakes1, i)

    return mistakes0, mistakes1


# My function
def load_data(file_name):
    """ Loads data from a .mat file. MATLAB 5.0 MAT-file
        Input: A .mat file name. The extension is not needed.
        Returns: neg_examples_nobias, pos_examples_nobias, w_init, w_gen_feas
    """
    mat_contents = sio.loadmat(file_name)
    neg_examples_nobias = mat_contents['neg_examples_nobias']
    pos_examples_nobias = mat_contents['pos_examples_nobias']
    w_init = mat_contents['w_init']
    w_gen_feas = mat_contents['w_gen_feas']

    return neg_examples_nobias, pos_examples_nobias, w_init, w_gen_feas



# There are 4 datasets: dataset1, ... , dataset4
# MATLAB 5.0 MAT-file, written by Octave 3.2.4, 2012-10-03
neg_examples_nobias, pos_examples_nobias, w_init, w_gen_feas = load_data(".\\Datasets\\dataset4")

learn_perceptron(neg_examples_nobias, pos_examples_nobias, w_init, w_gen_feas)
