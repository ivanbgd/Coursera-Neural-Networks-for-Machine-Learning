import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from collections import namedtuple
from sys import exit


def load_data(file_name = ".\\data.mat"):
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
    
    #print(data.dtype)                       # [('training', 'O'), ('validation', 'O'), ('test', 'O')]  -->
                                             # -->  dtype=[('inputs', 'O'), ('targets', 'O')]), 
                                             #      dtype=[('targets', 'O'), ('inputs', 'O')]),
                                             #      dtype=[('inputs', 'O'), ('targets', 'O')])

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


def model_to_theta(model):
    """
    This function takes a model (or gradient in model form), and turns it into one long vector. See also theta_to_model.
    Theta are model's parameters, that is, weights.
    """
    input_to_hid_transpose = model.input_to_hid.T    # (256, num_hid)
    hid_to_class_transpose = model.hid_to_class.T    # (num_hid, 10)
    # Fortran (Matlab) order! We can use 'C' in flatten() in this case (2-D) if we don't transpose them in the above two lines,
    # but this follows the original code, and is more general.
    theta = np.concatenate((input_to_hid_transpose.flatten('F'), hid_to_class_transpose.flatten('F')), axis=0)
    return theta


def theta_to_model(theta):
    """
    This function takes a model (or gradient) in the form of one long vector (maybe produced by model_to_theta),
    and restores it to the structure format, i.e. with fields .input_to_hid and .hid_to_class, both matrices.
    Theta are model's parameters, that is, weights.
    """
    Model = namedtuple('Model', ['input_to_hid', 'hid_to_class'])
    n_hid = theta.shape[0] // (256+10)
    input_to_hid = (theta[0 : 256 * n_hid].reshape((256, n_hid), order='F').copy()).T
    hid_to_class = (theta[256 * n_hid : theta.shape[0]].reshape((n_hid, 10), order='F').copy()).T
    model = Model(input_to_hid, hid_to_class)
    return model


def initial_model(n_hid):
    n_params = (256+10) * n_hid
    aux_array = np.linspace(0.0, (n_params - 1), n_params)
    as_row_vector = np.cos(aux_array)
    model = theta_to_model(as_row_vector * 0.1)    # We don't use random initialization for this assignment. This way, everybody will get the same results.
    return model


def logistic(input):
    return 1. / (1. + np.exp(-input))


def log_sum_exp_over_rows(a):
    """
    This computes log(sum(exp(a), 0)) in a numerically stable way
    """
    maxs_small = a.max(axis=0)
    maxs_big = np.tile(maxs_small, (a.shape[0], 1))
    ret = np.log(np.sum(np.exp(a - maxs_big), 0)) + maxs_small
    return ret


def loss(model, data, wd_coefficient):
    """
    model.input_to_hid is a matrix of size <number of hidden units> by <number of inputs i.e. 256>. It contains the weights from the input units to the hidden units.
    model.hid_to_class is a matrix of size <number of classes i.e. 10> by <number of hidden units>. It contains the weights from the hidden units to the softmax units.
    data.inputs is a matrix of size <number of inputs i.e. 256> by <number of data cases>. Each column describes a different data case. 
    data.targets is a matrix of size <number of classes i.e. 10> by <number of data cases>. Each column describes a different data case.
        It contains a one-of-N encoding of the class, i.e. one element in every column is 1 and the others are 0.
    """

    # Before we can calculate the loss, we need to calculate a variety of intermediate values, like the state of the hidden units.
    hid_input = np.dot(model.input_to_hid, data.inputs)     # input to the hidden units, i.e. before the logistic. size: <number of hidden units> by <number of data cases>
    hid_output = logistic(hid_input)                        # output of the hidden units, i.e. after the logistic. size: <number of hidden units> by <number of data cases>
    class_input = np.dot(model.hid_to_class, hid_output)    # input to the components of the softmax. size: <number of classes, i.e. 10> by <number of data cases>
  
    # The following three lines of code implement the softmax.
    # However, it's written differently from what the lectures say.
    # In the lectures, a softmax is described using an exponential divided by a sum of exponentials.
    # What we do here is exactly equivalent (you can check the math or just check it in practice), but this is more numerically stable. 
    # "Numerically stable" means that this way, there will never be really big numbers involved.
    # The exponential in the lectures can lead to really big numbers, which are fine in mathematical equations, but can lead to all sorts of problems in Octave.
    # Octave isn't well prepared to deal with really large numbers, like the number 10 to the power 1000. Computations with such numbers get unstable, so we avoid them.
    class_normalizer = log_sum_exp_over_rows(class_input)   # log(sum(exp of class_input)) is what we subtract to get properly normalized log class probabilities. size: <1> by <number of data cases>
    log_class_prob = class_input - np.tile(class_normalizer, (class_input.shape[0], 1))  # log of probability of each class. size: <number of classes, i.e. 10> by <number of data cases>
    class_prob = np.exp(log_class_prob)    # probability of each class. Each column (i.e. each case) sums to 1. size: <number of classes, i.e. 10> by <number of data cases>
    #print("class_prob:", class_prob.shape)
    #print("class_prob sum over rows:", np.sum(class_prob, axis=0))
  
    classification_loss = -np.mean(np.sum(log_class_prob * data.targets, axis=0))   # select the right log class probability using that sum; then take the mean over all data cases.
    wd_loss = 0.5 * wd_coefficient * np.sum(np.power(model_to_theta(model), 2))     # weight decay loss. very straightforward: E = 1/2 * wd_coeffecient * theta^2
    ret = classification_loss + wd_loss
    return ret


# OUR CODE HERE
def d_loss_by_d_model(model, data, wd_coefficient):
    """
    model.input_to_hid is a matrix of size <number of hidden units> by <number of inputs i.e. 256>
    model.hid_to_class is a matrix of size <number of classes i.e. 10> by <number of hidden units>
    data.inputs is a matrix of size <number of inputs i.e. 256> by <number of data cases>. Each column describes a different data case. 
    data.targets is a matrix of size <number of classes i.e. 10> by <number of data cases>. Each column describes a different data case.
        It contains a one-of-N encoding of the class, i.e. one element in every column is 1 and the others are 0.
    
    The returned object is supposed to be exactly like parameter <model>, i.e. it has fields ret.input_to_hid and ret.hid_to_class.
    However, the contents of those matrices are gradients (d loss by d model parameter), instead of model parameters.
     
    This is the only function that you're expected to change. Right now, it just returns a lot of zeros, which is obviously not the correct output.
    Your job is to replace that by a correct computation.
    """
    input_to_hid = model.input_to_hid * 0
    hid_to_class = model.hid_to_class * 0
    
    # OUR CODE HERE
    ret = None
    return ret


def test_gradient(model, data, wd_coefficient):
    base_theta = model_to_theta(model)
    h = 1e-2
    correctness_threshold = 1e-5
    analytic_gradient = model_to_theta(d_loss_by_d_model(model, data, wd_coefficient))

    # Test the gradient not for every element of theta, because that's a lot of work. Test for only a few elements.
    for i in range(100):
        test_index = (i * 1299721) % base_theta.shape[0] + 0    # 1299721 is prime and thus ensures a somewhat random-like selection of indices
        analytic_here = analytic_gradient[test_index]
        theta_step = np.zeros_like(base_theta)
        theta_step[test_index] = h
        contribution_distances = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
        contribution_weights = np.array([1/280, -4/105, 1/5, -4/5, 4/5, -1/5, 4/105, -1/280])
        temp = 0.0
        for contribution_index in range(8):
            temp = temp + loss(theta_to_model(base_theta + theta_step * contribution_distances[contribution_index]), data, wd_coefficient) * contribution_weights[contribution_index]
        fd_here = temp / h
        diff = np.abs(analytic_here - fd_here)
        #print('%d %e %e %e %e' % (test_index, base_theta[test_index], diff, fd_here, analytic_here))
        if (diff < correctness_threshold):
            continue
        if (diff / (np.abs(analytic_here) + np.abs(fd_here)) < correctness_threshold):
            continue
        exit("[ERROR] Theta element #{}, with value {:e}, has finite difference gradient {:e} but analytic gradient {:e}. That looks like an error.\n".\
            format(test_index, base_theta[test_index], fd_here, analytic_here))

    print("Gradient test passed. That means that the gradient that your code computed is within 0.001% of the gradient that the finite difference approximation computed,",
        "so the gradient calculation procedure is probably correct (not certainly, but probably).\n")


def classification_performance(model, data):
    """
    This returns the fraction of data cases that are incorrectly classified by the model.
    """
    hid_input = np.dot(model.input_to_hid, data.inputs)         # input to the hidden units, i.e. before the logistic. size: <number of hidden units> by <number of data cases>
    hid_output = logistic(hid_input)                            # output of the hidden units, i.e. after the logistic. size: <number of hidden units> by <number of data cases>
    class_input = np.dot(model.hid_to_class, hid_output)        # input to the components of the softmax. size: <number of classes, i.e. 10> by <number of data cases>
  
    choices = np.argmax(class_input, axis=0)    # choices is integer: the chosen class [0-9]
    targets = np.argmax(data.targets, axis=0)   # targets is integer: the target class [0-9]
    
    ret = np.mean(choices != targets)
    return ret


def a3(wd_coefficient, n_hid, n_iters, learning_rate, momentum_multiplier, do_early_stopping, mini_batch_size):
    """
    Input:
        wd_coeffecient: weight decay coefficient.
        n_hid: number of hidden units
    """
    model = initial_model(n_hid)
    training, validation, test = load_data()

    Datas = namedtuple('Datas', ['training', 'validation', 'test'])
    datas = Datas(training, validation, test)

    n_training_cases = datas.training.inputs.shape[1]     # 1000
    
    if n_iters != 0:
        test_gradient(model, datas.training, wd_coefficient)

    # optimization
    theta = model_to_theta(model)
    momentum_speed = np.zeros_like(theta)
    training_data_losses = []
    validation_data_losses = []

    best_so_far = {}

    if do_early_stopping:
        best_so_far["theta"] = -1   # this will be overwritten soon
        best_so_far["validation_loss"] = np.inf
        best_so_far["after_n_iters"] = -1

    for optimization_iteration_i in range(n_iters):
        model = theta_to_model(theta)

        training_batch_start = (optimization_iteration_i * mini_batch_size) % n_training_cases + 0
        training_batch_inputs = datas.training.inputs[:, training_batch_start : training_batch_start + mini_batch_size - 0]
        training_batch_targets = datas.training.targets[:, training_batch_start : training_batch_start + mini_batch_size - 0]
        TrainingBatch = namedtuple('TrainingBatch', ['inputs', 'targets'])
        training_batch = TrainingBatch(training_batch_inputs, training_batch_targets)

        gradient = model_to_theta(d_loss_by_d_model(model, training_batch, wd_coefficient))
        momentum_speed = momentum_speed * momentum_multiplier - gradient
        theta = theta + momentum_speed * learning_rate

        model = theta_to_model(theta)
        training_data_losses.append(loss(model, datas.training, wd_coefficient))
        validation_data_losses.append(loss(model, datas.validation, wd_coefficient))

        if do_early_stopping and (validation_data_losses[-1] < best_so_far["validation_loss"]):
            best_so_far["theta"] = theta    # this will be overwritten soon
            best_so_far["validation_loss"] = validation_data_losses[-1]
            best_so_far["after_n_iters"] = optimization_iteration_i

        if (1 + optimization_iteration_i) % np.round(n_iters / 10) == 0:
            print('After {} optimization iterations, training data loss is {}, and validation data loss is {}'.\
                format(1 + optimization_iteration_i, training_data_losses[-1], validation_data_losses[-1]))

    if n_iters != 0:
        # check again, this time with more typical parameters
        test_gradient(model, datas.training, wd_coefficient)
    
    if do_early_stopping:
        print('Early stopping: validation loss was lowest after {} iterations. We chose the model that we had then.'.format(best_so_far["after_n_iters"]))
        theta = best_so_far["theta"]
    
    # The optimization is finished. Now do some reporting.

    model = theta_to_model(theta)

    if n_iters != 0:
        f = plt.figure(1)
        plt.clf()
        plt.plot(range(n_iters), training_data_losses, 'b')
        plt.plot(range(n_iters), validation_data_losses, 'r')
        plt.legend(['training', 'validation'])
        plt.ylabel('loss')
        plt.xlabel('iteration number')
        plt.show()

    datas2 = [datas.training, datas.validation, datas.test]
    data_names = ['training', 'validation', 'test']
    for data_i, data in enumerate(datas2):
        data_name = data_names[data_i]
        print('The loss on the {} data is {}.'.format(data_name, loss(model, data, wd_coefficient)))
        if wd_coefficient != 0:
            print('The classification loss (i.e. without weight decay) on the {} data is {}.'.format(data_name, loss(model, data, 0)))
        print('The classification error rate on the {} data is {}.\n'.format(data_name, classification_performance(model, data)))
    print()



# Questions 1 & 2
# A test run without any training:
try:
    a3(wd_coefficient = 0, n_hid = 0, n_iters = 0, learning_rate = 0, momentum_multiplier = 0, do_early_stopping = False, mini_batch_size = 0)
except Exception as e:
    print("[ERROR] Test run failed!\n{}\n".format(e))

# Question 3
a3(wd_coefficient = 1e7, n_hid = 7, n_iters = 10, learning_rate = 0, momentum_multiplier = 0, do_early_stopping = False, mini_batch_size = 4)
a3(wd_coefficient = 0, n_hid = 7, n_iters = 10, learning_rate = 0, momentum_multiplier = 0, do_early_stopping = False, mini_batch_size = 4)
a3(wd_coefficient = 0, n_hid = 10, n_iters = 70, learning_rate = 0.005, momentum_multiplier = 0, do_early_stopping = False, mini_batch_size = 4)

# Questions 4 & 5
a3(wd_coefficient = 0, n_hid = 10, n_iters = 70, learning_rate = 0.5, momentum_multiplier = 0, do_early_stopping = False, mini_batch_size = 4)
for momentum in [0.0, 0.9]:
    for learning_rate in [0.002, 0.01, 0.05, 0.2, 1.0, 5.0, 20.0]:
        print('\n')
        print('*****************************************************')
        print('\tmomentum = {}, learning_rate = {}'.format(momentum, learning_rate))
        print('*****************************************************')
        a3(wd_coefficient = 0, n_hid = 10, n_iters = 70, learning_rate = learning_rate, momentum_multiplier = momentum, do_early_stopping = False, mini_batch_size = 4)

# Question 6
a3(wd_coefficient = 0, n_hid = 200, n_iters = 1000, learning_rate = 0.35, momentum_multiplier = 0.9, do_early_stopping = False, mini_batch_size = 100)
#a3(wd_coefficient = 0, n_hid = 200, n_iters = 175, learning_rate = 0.35, momentum_multiplier = 0.9, do_early_stopping = False, mini_batch_size = 100)  # my variant

# Question 7
a3(wd_coefficient = 0, n_hid = 200, n_iters = 1000, learning_rate = 0.35, momentum_multiplier = 0.9, do_early_stopping = True, mini_batch_size = 100)

# Question 8
for wd in [0, 0.0001, 0.001, 0.1, 1, 10]:
    print('\n')
    print('************************************************************************')
    print('  wd_coefficient = {}, momentum = {}, learning_rate = {}'.format(wd, 0.9, 0.35))
    print('************************************************************************')
    a3(wd_coefficient = wd, n_hid = 200, n_iters = 1000, learning_rate = 0.35, momentum_multiplier = 0.9, do_early_stopping = False, mini_batch_size = 100)

# Question 9
for n_hid in [10, 30, 100, 130, 200]:
    print('\n')
    print('************************************************************************')
    print('\tn_hid = {}, momentum = {}, learning_rate = {}'.format(n_hid, 0.9, 0.35))
    print('************************************************************************')
    a3(wd_coefficient = 0, n_hid = n_hid, n_iters = 1000, learning_rate = 0.35, momentum_multiplier = 0.9, do_early_stopping = False, mini_batch_size = 100)

# Question 10
for n_hid in [18, 37, 83, 113, 236]:
    print('\n')
    print('************************************************************************')
    print('\tn_hid = {}, momentum = {}, learning_rate = {}'.format(n_hid, 0.9, 0.35))
    print('************************************************************************')
    a3(wd_coefficient = 0, n_hid = n_hid, n_iters = 1000, learning_rate = 0.35, momentum_multiplier = 0.9, do_early_stopping = True, mini_batch_size = 100)




