import numpy as np
import scipy.io as sio
from fprop import fprop


def load_data(N = 100, file_name = ".\\data.mat"):
    """ Loads data from a .mat file. MATLAB 5.0 MAT-file
        This method loads the training, validation and test sets.
        It also divides the training set into mini-batches.
        Inputs:
            N: Mini-batch size. Default = 100.
            file_name: A .mat file name. The extension is not needed.
        Returns:
            train_input: An array of size D X N X M, where
                D: number of input dimensions (in this case, 3).
                N: size of each mini-batch (in this case, 100).
                M: number of minibatches.
            train_target: An array of size 1 X N X M.
            valid_input: An array of size D X number of points in the validation set.
            test: An array of size D X number of points in the test set.
            vocab: Vocabulary containing index to word mapping. vocab[i][0] gives the element (a string).
    """
    mat_contents = sio.loadmat(file_name)
    data = mat_contents['data']

    #print(data.dtype)                       # [('testData', 'O'), ('trainData', 'O'), ('validData', 'O'), ('vocab', 'O')]
    #print(data['trainData'][0][0].shape)    # (4, 372550)
    #print(data['validData'][0][0].shape)    # (4, 46568)
    #print(data['testData'][0][0].shape)     # (4, 46568)
    #print(data['vocab'][0][0].shape)        # (1, 250)
    
    numdims = data['trainData'][0][0].shape[0]  # 4
    D = numdims - 1                             # 3
    M = data['trainData'][0][0].shape[1] // N   # 3725
    train_input = np.reshape(data['trainData'][0][0][:D, :N*M], (D, N, M))  # (3, 100, 3725), int32
    train_target = np.reshape(data['trainData'][0][0][D, :N*M], (1, N, M))  # (1, 100, 3725), int32
    valid_input = data['validData'][0][0][:D, :]                            # (3, 46568), int32
    valid_target = data['validData'][0][0][D, :]                            # (46568,), int32
    test_input = data['testData'][0][0][:D, :]                              # (3, 46568), int32
    test_target = data['testData'][0][0][D, :]                              # (46568,), int32
    vocab = data['vocab'][0][0][0]      # (250,), object (which is an ndarray whose elements are strings of type '<Ux', where x is number of Unicode characters in a string (that element))

    # Data is in Matlab format, which indexes from 1. Python indexing starts at 0.
    train_input -= 1
    train_target -= 1
    valid_input -= 1
    valid_target -= 1
    test_input -= 1
    test_target -= 1
    
    return train_input, train_target, valid_input, valid_target, test_input, test_target, vocab


def display_nearest_words(word, model, k):
    """
    Shows the k-nearest words to the query word.
    Inputs:
        word: The query word as a string.
        model: Model returned by the training script.
        k: The number of nearest words to display.
    Example usage:
        display_nearest_words('school', model, 10)
    """

    word_embedding_weights = model["word_embedding_weights"]
    vocab = model["vocab"]      # (250,)
    vocab = list(vocab)
    try:
        id = vocab.index(word)
    except ValueError:
        print("Word '{}' not in vocabulary.\n".format(word))
        return

    # Compute distance to every other word.
    vocab_size = len(vocab)     # 250
    word_rep = word_embedding_weights[id, :]
    diff = word_embedding_weights - np.tile(word_rep, (vocab_size, 1))
    distance = np.sqrt(np.sum(diff * diff, 1))      # (250,)

    # Sort by distance.
    #distance_sorted = np.sort(distance)
    order = np.argsort(distance)
    order = order[1 : k+1]  # The nearest word is the query word itself, skip that. (10,)

    print("Your word is:", word)
    print("The most similar {} words are:".format(k))
    for i in range(k):
        print(" {} {:.2f}".format(vocab[order[i]], distance[order[i]]))
        #print(" {} {:.2f}".format(vocab[order[i]], distance_sorted[i+1]))
    print("")


def word_distance(word1, word2, model):
    """
    Shows the L2 distance between word1 and word2 in the word_embedding_weights.
    Inputs:
        word1: The first word as a string.
        word2: The second word as a string.
        model: Model returned by the training script.
    Example usage:
        word_distance('school', 'university', model)
    """
    word_embedding_weights = model["word_embedding_weights"]
    vocab = model["vocab"]      # (250,)
    vocab = list(vocab)

    try:
        id1 = vocab.index(word1)
    except ValueError:
        print("Word '{}' not in vocabulary.\n".format(word1))
        return

    try:
        id2 = vocab.index(word2)
    except ValueError:
        print("Word '{}' not in vocabulary.\n".format(word2))
        return

    word_rep1 = word_embedding_weights[id1, :]
    word_rep2 = word_embedding_weights[id2, :]
    diff = word_rep1 - word_rep2
    distance = np.sqrt(np.sum(diff * diff))

    return distance


def predict_next_word(word1, word2, word3, model, k):
    """
    Predicts the next word.
    Inputs:
        word1: The first word as a string.
        word2: The second word as a string.
        word3: The third word as a string.
        model: Model returned by the training script.
        k: The k most probable predictions are shown.
    Example usage:
        predict_next_word('john', 'might', 'be', model, 3)
        predict_next_word('life', 'in', 'new', model, 3)
    """
    word_embedding_weights = model["word_embedding_weights"]
    vocab = model["vocab"]      # (250,)
    vocab = list(vocab)

    try:
        id1 = vocab.index(word1)
    except ValueError:
        print("Word '{}' not in vocabulary.\n".format(word1))
        return

    try:
        id2 = vocab.index(word2)
    except ValueError:
        print("Word '{}' not in vocabulary.\n".format(word2))
        return

    try:
        id3 = vocab.index(word3)
    except ValueError:
        print("Word '{}' not in vocabulary.\n".format(word3))
        return

    input = np.array([id1, id2, id3])       # (3,)
    input = np.expand_dims(input, axis=1)   # (3, 1)
    embedding_layer_state, hidden_layer_state, output_layer_state = \
            fprop(input, model["word_embedding_weights"], model["embed_to_hid_weights"], model["hid_to_output_weights"], model["hid_bias"], model["output_bias"])
    prob = np.sort(output_layer_state, axis=None)[::-1]     # (250,); output_layer_state.shape is (250, 1).
    indices = np.argsort(-output_layer_state, axis=None)

    for i in range(k):
        print("{} {} {} {}    Prob: {:.5f}".format(word1, word2, word3, vocab[int(indices[i])], prob[i+1]))
    print("")

