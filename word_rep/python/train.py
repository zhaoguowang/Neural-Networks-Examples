#!/usr/bin/env python

# import modules used here -- sys is a very standard one
import sys
import numpy as np
import scipy.io as sio

from load import load

# SET HYPERPARAMETERS HERE.
batchsize = 100  # Mini-batch size.
learning_rate = 0.1  # Learning rate; default = 0.1.
momentum = 0.9  # Momentum; default = 0.9.
numhid1 = 50  # Dimensionality of embedding space; default = 50.
numhid2 = 200  # Number of units in hidden layer; default = 200.
init_wt = 0.01  # Standard deviation of the normal distribution
                 # which is sampled to get the initial weights; default = 0.01

# VARIABLES FOR TRACKING TRAINING PROGRESS.
show_training_CE_after = 100
show_validation_CE_after = 1000


# Load Data
[train_input, train_target, valid_input, valid_target, \
  test_input, test_target, vocab] = load(batchsize)

[numwords, batchsize, numbatches] = train_input.shape 
vocab_size = vocab.size

word_embedding_weights = init_wt * np.random.rand(vocab_size, numhid1)
embed_to_hid_weights = init_wt * np.random.rand(numwords * numhid1, numhid2)
hid_to_output_weights = init_wt * np.random.rand(numhid2, vocab_size)

hid_bias = np.zeros((numhid2, 1))
output_bias = np.zeros((vocab_size, 1))


# Gather our code in a main() function
def main():
	print output_bias.shape
    #print 'Hello there', numwords, batchsize, numbatches, vocab_size, 'word_embedding_weights.Shape ', word_embedding_weights.shape

# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()