import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from collections import namedtuple
from sys import exit

_report_calls_to_sample_bernoulli = True


def load_data(file_name = ".\\data_set.mat"):
    """ Loads data from a .mat file. MATLAB 5.0 MAT-file
        This method loads the training, validation and test sets.
        They are already split in the data file.
        Columns represent data samples, where rows contain individual pixels or one-hot encoded classes (target digits).
        Input:
            file_name: A .mat file name. The extension is not needed.
        Returns:
            training: A namedtuple of training 'inputs' and 'targets'. Respective shapes are: (256, 1000), (10, 1000).
            validation: A namedtuple of validation 'inputs' and 'targets'. Respective shapes are: (256, 1000), (10, 1000).
            test: A namedtuple of test 'inputs' and 'targets'. Respective shapes are: (256, 9000), (10, 9000).
    """
    mat_contents = sio.loadmat(file_name)
    data = mat_contents['data']

    #print(mat_contents)
    #print(data)
    #print(data.dtype)                       # [('training', 'O'), ('validation', 'O'), ('test', 'O')]  -->
                                             # -->  dtype=[('inputs', 'O'), ('targets', 'O')]),  dtype=[('targets', 'O'), ('inputs', 'O')]),  dtype=[('inputs', 'O'), ('targets', 'O')])

    #print(data['training'][0][0][0][0][0].shape)        # (256, 1000)
    #print(data['training'][0][0][0][0][1].shape)        # (10, 1000)
    #print(data['validation'][0][0][0][0][0].shape)      # (10, 1000)
    #print(data['validation'][0][0][0][0][1].shape)      # (256, 1000)
    #print(data['test'][0][0][0][0][0].shape)            # (256, 9000)
    #print(data['test'][0][0][0][0][1].shape)            # (10, 9000)

    train_input = data['training'][0][0][0][0][0]           # (256, 1000)
    train_target = data['training'][0][0][0][0][1]          # (10, 1000)
    valid_input = data['validation'][0][0][0][0][1]         # (256, 1000)
    valid_target = data['validation'][0][0][0][0][0]        # (10, 1000)
    test_input = data['test'][0][0][0][0][0]                # (256, 9000)
    test_target = data['test'][0][0][0][0][1]               # (10, 9000)

    Train = namedtuple('Train', ['inputs', 'targets'])
    Valid = namedtuple('Valid', ['inputs', 'targets'])
    Test = namedtuple('Test', ['inputs', 'targets'])

    training = Train(train_input, train_target)
    validation = Valid(valid_input, valid_target)
    test = Test(test_input, test_target)
    
    return training, validation, test


def a4_rand(requested_size, seed):
    """
    Returns 'random' data of the requested shape.
    The data are sampled from the file 'a4_randomness_source.mat'.
    Seed is used to calculate the starting point in the file.
    """
    mat_contents = sio.loadmat(".\\a4_randomness_source.mat")
    randomness_source = mat_contents['randomness_source']   # (1, 350381)

    requested_size = list(requested_size)
    start_i = int(np.round(seed)) % int(np.round(randomness_source.shape[1] / 10)) + 0
    if (start_i + np.prod(requested_size)) >= (randomness_source.shape[1] + 0):
        exit('a4_rand failed to generate an array of that size (too big)')
    ret = np.reshape(randomness_source[:, start_i : start_i+np.prod(requested_size)-0], newshape=tuple(requested_size), order='F')
    return ret


def log_sum_exp_over_rows(a):
    """
    This computes log(sum(exp(a), 0)) in a numerically stable way.
    """
    maxs_small = a.max(axis=0)
    maxs_big = np.tile(maxs_small, (a.shape[0], 1))
    ret = np.log(np.sum(np.exp(a - maxs_big), 0)) + maxs_small
    return ret


def classification_phi_gradient(input_to_class, data):
    """
    This is about a very simple model: there's an input layer, and a softmax output layer. There are no hidden layers, and no biases.
    This returns the gradient of phi (a.k.a. negative the loss) for the <input_to_class> matrix.
    <input_to_class> is a matrix of size <number of classes> by <number of input units>.
    <data> has fields .inputs (matrix of size <number of input units> by <number of data cases>) and .targets (matrix of size <number of classes> by <number of data cases>).
    """
    # first: forward pass
    class_input = np.dot(input_to_class, data.inputs)       # input to the components of the softmax. size: <number of classes> by <number of data cases>
    class_normalizer = log_sum_exp_over_rows(class_input)   # log(sum(exp)) is what we subtract to get normalized log class probabilities. size: <1> by <number of data cases>
    log_class_prob = class_input - np.tile(class_normalizer, (class_input.shape[0], 1)) # log of probability of each class. size: <number of classes> by <number of data cases>
    class_prob = np.exp(log_class_prob)                     # probability of each class. Each column (i.e. each case) sums to 1. size: <number of classes> by <number of data cases>
    # now: gradient computation
    d_loss_by_d_class_input = -(data.targets - class_prob) / data.inputs.shape[1]   # size: <number of classes> by <number of data cases>
    d_loss_by_d_input_to_class = np.dot(d_loss_by_d_class_input, data.inputs.T)     # size: <number of classes> by <number of input units>
    d_phi_by_d_input_to_class = -d_loss_by_d_input_to_class
    return d_phi_by_d_input_to_class

 
def argmax_over_rows(matrix):
    indices = np.argmax(matrix, axis=0)
    return indices


def describe_matrix(matrix):
    print('Describing a matrix of size {} by {}. The mean of the elements is {}. The sum of the elements is {}.'\
          .format(matrix.shape[0], matrix.shape[1], np.mean(matrix), np.sum(matrix)))


def extract_mini_batch(data_set, start_i, n_cases):
    """
    Use Python indexing for start_i, i.e. indexing that starts from 0.
    """
    mini_batch_inputs = data_set.inputs[:, start_i : start_i + n_cases - 0]
    mini_batch_targets = data_set.targets[:, start_i : start_i + n_cases - 0]
    Mini_batch = namedtuple('Mini_batch', ['inputs', 'targets'])
    mini_batch = Mini_batch(mini_batch_inputs, mini_batch_targets)
    return mini_batch


def logistic(input):
    return 1. / (1. + np.exp(-input))


def optimize(model_shape, gradient_function, training_data, learning_rate, n_iterations):
    """
    This trains a model that's defined by a single matrix of weights.
    <model_shape> is the shape of the array of weights.
    <gradient_function> is a function that takes parameters <model> and <data> and returns the gradient (or approximate gradient in the case of CD-1) of the function that we're maximizing.
        Note the contrast with the loss function that we saw in PA3, which we were minimizing. The returned gradient is an array of the same shape as the provided <model> parameter.
    This uses mini-batches of size 100, momentum of 0.9, no weight decay, and no early stopping.
    This returns the matrix of weights of the trained model.
    """
    model = (a4_rand(model_shape, np.prod(model_shape)) * 2 - 1) * 0.1
    momentum_speed = np.zeros(model_shape)
    mini_batch_size = 100
    start_of_next_mini_batch = 1 - 1
    for iteration_number in range(n_iterations):
        mini_batch = extract_mini_batch(training_data, start_of_next_mini_batch, mini_batch_size)
        start_of_next_mini_batch = (start_of_next_mini_batch + mini_batch_size) % training_data.inputs.shape[1]
        gradient = gradient_function(model, mini_batch)
        momentum_speed = 0.9 * momentum_speed + gradient    # ascent
        model = model + momentum_speed * learning_rate
    return model


def show_rbm(rbm_w):
    n_hid = rbm_w.shape[0]
    n_rows = int(np.ceil(np.sqrt(n_hid)))
    blank_lines = 4
    distance = 16 + blank_lines
    to_show = np.zeros([n_rows * distance + blank_lines, n_rows * distance + blank_lines])
    for i in range(n_hid):
        row_i = int(np.floor(i / n_rows))
        col_i = i % n_rows
        pixels = np.reshape(rbm_w[i+0, :], [16, 16]).T
        row_base = row_i*distance + blank_lines
        col_base = col_i*distance + blank_lines
        to_show[row_base+0:row_base+16, col_base+0:col_base+16] = pixels
    extreme = np.max(np.abs(to_show))
    try:
        plt.imshow(to_show, cmap='gray')
        plt.title('Hidden units of the RBM')
        plt.show()
    except:
        print('Failed to display the RBM. No big deal (you do not need the display to finish the assignment), but you are missing out on an interesting picture.');
    return


def _sample_bernoulli(probabilities):
    """
    Returns a "binary" matrix of the same shape as 'probabilities'.
    It is binary in the sense that it contains only 0s and 1s.
    This creates a Bernoulli distribution over samples from the file 'a4_randomness_source.mat'.
    """
    if _report_calls_to_sample_bernoulli:
        print('_sample_bernoulli() was called with a matrix of size {} by {}.'.format(probabilities.shape[0], probabilities.shape[1]))
    seed = np.sum(probabilities)
    binary = 1 * (probabilities > a4_rand(probabilities.shape, seed))   # The "1*" is to avoid the "logical" data type, which just confuses things.
    return binary


######################
### OUR CODE BELOW ###
######################

def visible_state_to_hidden_probabilities(rbm_w, visible_state):
    """
    <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
    <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
    The returned value is a matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
    This takes in the (binary) states of the visible units, and returns the activation probabilities of the hidden units conditional on those states.
    """
    raise NotImplementedError
    return hidden_probability


def hidden_state_to_visible_probabilities(rbm_w, hidden_state):
    """
    <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
    <hidden_state> is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
    The returned value is a matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
    This takes in the (binary) states of the hidden units, and returns the activation probabilities of the visible units, conditional on those states.
    """
    raise NotImplementedError
    return visible_probability


def configuration_goodness(rbm_w, visible_state, hidden_state):
    """
    <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
    <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
    <hidden_state> is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
    This returns a scalar: the mean over cases of the goodness (negative energy) of the described configurations.
    """
    raise NotImplementedError
    return G


def configuration_goodness_gradient(visible_state, hidden_state):
    """
    <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
    <hidden_state> is a (possibly but not necessarily binary) matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
    You don't need the model parameters for this computation.
    This returns the gradient of the mean configuration goodness (negative energy, as computed by function <configuration_goodness>) with respect to the model parameters.
        Thus, the returned value is of the same shape as the model parameters, which by the way are not provided to this function.
        Notice that we're talking about the mean over data cases (as opposed to the sum over data cases).
    """
    raise NotImplementedError
    return d_G_by_rbm_w


def cd1(rbm_w, visible_data):
    """
    This is an implementation of Contrastive Divergence gradient estimator with 1 full Gibbs update, a.k.a. CD-1.
    <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
    <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
    The returned value is the gradient approximation produced by CD-1 (Contrastive Divergence 1). It's of the same shape as <rbm_w>.
    """
    raise NotImplementedError
    return ret


