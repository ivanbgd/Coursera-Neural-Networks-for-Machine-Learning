import numpy as np
import sys
from time import time
from utils import *
from fprop import fprop

def train(epochs = 1):
    """ This function trains a neural network language model.
        Inputs:
          epochs: Number of epochs to run.
        Output:
          model: A struct containing the learned weights and biases and vocabulary.
    """
    start_time = time()
    
    # SET HYPERPARAMETERS HERE.
    batchsize = 100         # Mini-batch size.
    learning_rate = 0.1     # Learning rate, default = 0.1.
    momentum = 0.9          # Momentum, default = 0.9.
    numhid1 = 50            # Dimensionality of embedding space, default = 50.
    numhid2 = 200           # Number of units in hidden layer, default = 200.
    init_wt = 0.01          # Standard deviation of the normal distribution which is sampled to get the initial weights, default = 0.01

    # VARIABLES FOR TRACKING TRAINING PROGRESS.
    show_training_CE_after = 100
    show_validation_CE_after = 1000

    # LOAD DATA.
    train_input, train_target, valid_input, valid_target, test_input, test_target, vocab = load_data(batchsize)
    numwords, batchsize, numbatches = train_input.shape                     # 3, 100, 3725
    vocab_size = vocab.shape[0]                                             # 250   vocab_size = size(vocab, 2);

    # INITIALIZE WEIGHTS AND BIASES.
    word_embedding_weights = init_wt * np.random.randn(vocab_size, numhid1)         # (250, 50)
    embed_to_hid_weights = init_wt * np.random.randn(numwords * numhid1, numhid2)   # (150, 200)
    hid_to_output_weights = init_wt * np.random.randn(numhid2, vocab_size)          # (200, 250)
    hid_bias = np.zeros((numhid2, 1))                                               # (200, 1)
    output_bias = np.zeros((vocab_size, 1))                                         # (250, 1)

    word_embedding_weights_delta = np.zeros((vocab_size, numhid1))          # (250, 50)
    word_embedding_weights_gradient = np.zeros((vocab_size, numhid1))       # (250, 50)
    embed_to_hid_weights_delta = np.zeros((numwords * numhid1, numhid2))    # (150, 200)
    hid_to_output_weights_delta = np.zeros((numhid2, vocab_size))           # (200, 250)
    hid_bias_delta = np.zeros((numhid2, 1))                                 # (200, 1)
    output_bias_delta = np.zeros((vocab_size, 1))                           # (250, 1)
    expansion_matrix = np.eye((vocab_size))                                 # (250, 250)
    count = 0
    tiny = np.exp(-30)

    # TRAIN.
    for epoch in range(1, epochs + 1):
        print('Epoch {}'.format(epoch))
        this_chunk_CE = 0
        trainset_CE = 0
        # LOOP OVER MINI-BATCHES.
        for m in range(1, numbatches + 1):
            input_batch = train_input[:, :, m - 1]                              # (3, 100)
            target_batch = train_target[:, :, m - 1]                            # (1, 100)
            #print("input_batch:", input_batch.shape)
            #print("target_batch:", target_batch.shape)

            # FORWARD PROPAGATE.
            # Compute the state of each layer in the network given the input batch
            # and all weights and biases
            embedding_layer_state, hidden_layer_state, output_layer_state = fprop(
                    input_batch,
                    word_embedding_weights, embed_to_hid_weights,
                    hid_to_output_weights, hid_bias, output_bias)

            # COMPUTE DERIVATIVE.
            ## Expand the target to a sparse 1-of-K vector.
            expanded_target_batch = expansion_matrix[:, target_batch]               # (250, 1, 100)
            expanded_target_batch = expanded_target_batch.reshape(vocab_size, -1)   # (250, 100)
            #print("expanded_target_batch:", expanded_target_batch.shape)
            ## Compute derivative of cross-entropy loss function.
            error_deriv = output_layer_state - expanded_target_batch                # (250, 100)
            #print("error_deriv:", error_deriv.shape)
            
            # MEASURE LOSS FUNCTION.
            CE = -np.sum(np.sum(expanded_target_batch * np.log(output_layer_state + tiny))) / batchsize
            count = count + 1
            this_chunk_CE = this_chunk_CE + (CE - this_chunk_CE) / count
            trainset_CE = trainset_CE + (CE - trainset_CE) / m
            if (np.mod(m, show_training_CE_after) == 0):
                print('Batch {} Train CE {:.3f}'.format(m, this_chunk_CE))
                count = 0
                this_chunk_CE = 0
            sys.stdout.flush()

            # BACK PROPAGATE.
            ## OUTPUT LAYER.
            hid_to_output_weights_gradient = np.dot(hidden_layer_state, error_deriv.T)                                                  # (200, 250)
            output_bias_gradient = np.sum(error_deriv, axis=1)                                                                          # (250,)
            output_bias_gradient = output_bias_gradient[:, np.newaxis]                                                                  # (250, 1)
            #output_bias_gradient = output_bias_gradient.reshape(output_bias_gradient.shape[0], 1)                                      # (250, 1)
            back_propagated_deriv_1 = np.dot(hid_to_output_weights, error_deriv) * hidden_layer_state * (1.0 - hidden_layer_state)      # (200, 100)
            #print("hid_to_output_weights_gradient:", hid_to_output_weights_gradient.shape)
            #print("output_bias_gradient:", output_bias_gradient.shape)
            #print("back_propagated_deriv_1:", back_propagated_deriv_1.shape)

            ## HIDDEN LAYER.
            # FILL IN CODE. Replace the line below by one of the options.
            #embed_to_hid_weights_gradient = np.zeros((numhid1 * numwords, numhid2))
            # Options:
            # (a) embed_to_hid_weights_gradient = np.dot(back_propagated_deriv_1.T, embedding_layer_state)
            # (b) embed_to_hid_weights_gradient = np.dot(embedding_layer_state, back_propagated_deriv_1.T)
            # (c) embed_to_hid_weights_gradient = back_propagated_deriv_1
            # (d) embed_to_hid_weights_gradient = embedding_layer_state
            #print("embed_to_hid_weights_gradient:", embed_to_hid_weights_gradient.shape)                   # (150, 200)

            # FILL IN CODE. Replace the line below by one of the options.
            #hid_bias_gradient = np.zeros((numhid2, 1))
            # Options
            # (a) hid_bias_gradient = np.sum(back_propagated_deriv_1, 1)
            # (b) hid_bias_gradient = np.sum(back_propagated_deriv_1, 0)
            # (c) hid_bias_gradient = back_propagated_deriv_1
            # (d) hid_bias_gradient = back_propagated_deriv_1.T
            # Shape is (200,).
            hid_bias_gradient = np.expand_dims(hid_bias_gradient, axis=1)                                   # (200, 1)
            #hid_bias_gradient = hid_bias_gradient.reshape(hid_bias_gradient.shape[0], 1)                   # (200, 1)
            #print("hid_bias_gradient:", hid_bias_gradient.shape)
            
            # FILL IN CODE. Replace the line below by one of the options.
            #back_propagated_deriv_2 = np.zeros((numhid2, batchsize))                                       # (200, 100)
            # Options
            # (a) back_propagated_deriv_2 = np.dot(embed_to_hid_weights, back_propagated_deriv_1)
            # (b) back_propagated_deriv_2 = np.dot(back_propagated_deriv_1, embed_to_hid_weights)
            # (c) back_propagated_deriv_2 = np.dot(back_propagated_deriv_1.T, embed_to_hid_weights)
            # (d) back_propagated_deriv_2 = np.dot(back_propagated_deriv_1, embed_to_hid_weights.T)
            #print("back_propagated_deriv_2:", back_propagated_deriv_2.shape)                   # (150,100)

            word_embedding_weights_gradient[:] = 0

            ## EMBEDDING LAYER.
            for w in range(1, numwords + 1):
                #print(expansion_matrix[:, input_batch[w - 1, :]].shape)                        # (250, 100)
                #print(back_propagated_deriv_2[0 + (w - 1) * numhid1 : w * numhid1, :].shape)   # (50, 100)
                word_embedding_weights_gradient = word_embedding_weights_gradient + \
                    np.dot(expansion_matrix[:, input_batch[w - 1, :]], back_propagated_deriv_2[(w - 1) * numhid1 : w * numhid1, :].T)
            #print("word_embedding_weights_gradient:", word_embedding_weights_gradient.shape)   # (250, 50)

            # UPDATE WEIGHTS AND BIASES.
            word_embedding_weights_delta = momentum * word_embedding_weights_delta + word_embedding_weights_gradient / batchsize
            word_embedding_weights = word_embedding_weights - learning_rate * word_embedding_weights_delta

            embed_to_hid_weights_delta = momentum * embed_to_hid_weights_delta + embed_to_hid_weights_gradient / batchsize
            embed_to_hid_weights = embed_to_hid_weights - learning_rate * embed_to_hid_weights_delta

            hid_to_output_weights_delta = momentum * hid_to_output_weights_delta + hid_to_output_weights_gradient / batchsize
            hid_to_output_weights = hid_to_output_weights - learning_rate * hid_to_output_weights_delta

            hid_bias_delta = momentum * hid_bias_delta + hid_bias_gradient / batchsize              # (200, 1)
            hid_bias = hid_bias - learning_rate * hid_bias_delta                                    # (200, 1)

            output_bias_delta = momentum * output_bias_delta + output_bias_gradient / batchsize     # (250, 1)
            output_bias = output_bias - learning_rate * output_bias_delta                           # (250, 1)

            # VALIDATE.
            if (np.mod(m, show_validation_CE_after) == 0):
                print('Running validation ...')
                sys.stdout.flush()
          
                embedding_layer_state, hidden_layer_state, output_layer_state = fprop(valid_input, word_embedding_weights, embed_to_hid_weights, hid_to_output_weights, hid_bias, output_bias)
                datasetsize = valid_input.shape[1]
                expanded_valid_target = expansion_matrix[:, valid_target]
                CE = -np.sum(np.sum(expanded_valid_target * np.log(output_layer_state + tiny))) / datasetsize
                print(' Validation CE {:.3f}'.format(CE))
                sys.stdout.flush()

        print('  Average Training CE {:.3f}\n'.format(trainset_CE))
    
    print('Finished Training.')
    sys.stdout.flush()    
    print('Final Training CE {:.3f}'.format(trainset_CE))

    # EVALUATE ON VALIDATION SET.
    print('\nRunning validation ...')
    sys.stdout.flush()    
    embedding_layer_state, hidden_layer_state, output_layer_state = fprop(valid_input, word_embedding_weights, embed_to_hid_weights, hid_to_output_weights, hid_bias, output_bias)
    datasetsize = valid_input.shape[1]
    expanded_valid_target = expansion_matrix[:, valid_target]
    CE = -np.sum(np.sum(expanded_valid_target * np.log(output_layer_state + tiny))) / datasetsize
    print('Final Validation CE {:.3f}'.format(CE))
    sys.stdout.flush()
    
    # EVALUATE ON TEST SET.
    print('\nRunning test ...')
    sys.stdout.flush()    
    embedding_layer_state, hidden_layer_state, output_layer_state = fprop(test_input, word_embedding_weights, embed_to_hid_weights, hid_to_output_weights, hid_bias, output_bias)
    datasetsize = test_input.shape[1]
    expanded_test_target = expansion_matrix[:, test_target]
    CE = -np.sum(np.sum(expanded_test_target * np.log(output_layer_state + tiny))) / datasetsize
    print('Final Test CE {:.3f}'.format(CE))
    sys.stdout.flush()
    
    #model = [word_embedding_weights, embed_to_hid_weights, hid_to_output_weights, hid_bias, output_bias, vocab]
    model = {"word_embedding_weights" : word_embedding_weights, "embed_to_hid_weights" : embed_to_hid_weights, "hid_to_output_weights" : hid_to_output_weights,
             "hid_bias" : hid_bias, "output_bias" : output_bias, "vocab" : vocab}

    #model.word_embedding_weights = word_embedding_weights
    #model.embed_to_hid_weights = embed_to_hid_weights
    #model.hid_to_output_weights = hid_to_output_weights
    #model.hid_bias = hid_bias
    #model.output_bias = output_bias
    #model.vocab = vocab

    end_time = time()
    diff = end_time - start_time
    print("\nTraining took {:.3f} seconds\n".format(diff))

    return model

