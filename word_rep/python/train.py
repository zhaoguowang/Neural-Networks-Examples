#!/usr/bin/env python

# import modules used here -- sys is a very standard one
import sys
import numpy as np
import scipy.io as sio

from load import load
from fprop import fprop

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

word_embedding_weights = init_wt * np.random.randn(vocab_size, numhid1)
embed_to_hid_weights = init_wt * np.random.randn(numwords * numhid1, numhid2)
hid_to_output_weights = init_wt * np.random.randn(numhid2, vocab_size)

#word_embedding_weights = init_wt * np.ones((vocab_size, numhid1))
#embed_to_hid_weights = init_wt * np.ones((numwords * numhid1, numhid2))
#hid_to_output_weights = init_wt * np.ones((numhid2, vocab_size))


hid_bias = np.zeros((numhid2, 1))
output_bias = np.zeros((vocab_size, 1))

word_embedding_weights_delta = np.zeros((vocab_size, numhid1))
word_embedding_weights_gradient = np.zeros((vocab_size, numhid1))

embed_to_hid_weights_delta = np.zeros((numwords * numhid1, numhid2))

hid_to_output_weights_delta = np.zeros((numhid2, vocab_size))

hid_bias_delta = np.zeros((numhid2, 1))

output_bias_delta = np.zeros((vocab_size, 1))

expansion_matrix = np.eye(vocab_size)

count = 0;

tiny = np.exp(-30);

def train(epochs):
	for epoch in range(1, epochs + 1):
		print 'Epoch --------', epoch, '----------' 
		this_chunk_CE = 0;
  		trainset_CE = 0;
  		for m in range(0, numbatches):
  			input_batch = train_input[:, :, m]
  			target_batch = train_target[:, :, m]
  			#print 'Batch', m
  			global word_embedding_weights
  			global word_embedding_weights_delta
  			global embed_to_hid_weights
  			global hid_to_output_weights
  			global hid_bias
  			global output_bias

  			[embedding_layer_state, hidden_layer_state, output_layer_state] = \
	        	fprop(input_batch, \
      	      		word_embedding_weights, \
      	      		embed_to_hid_weights, \
        	    	hid_to_output_weights, \
        	    	hid_bias, \
        	    	output_bias)
	        
  			#print 'xxxx', output_layer_state.shape, target_batch.shape
  			expanded_target_batch = expansion_matrix[:, target_batch.ravel()]
  			error_deriv = output_layer_state - expanded_target_batch
  			CE = -np.sum(np.sum(expanded_target_batch * np.log(output_layer_state + tiny))) / batchsize
  			global count
  			count = count + 1
  			this_chunk_CE = this_chunk_CE + (CE - this_chunk_CE) / count
			trainset_CE = trainset_CE + (CE - trainset_CE) / (m + 1)

			print '\rBatch', (m + 1), 'Train CE', this_chunk_CE, 		

			if (m + 1) % show_training_CE_after == 0:
				print ''
				count = 0
				this_chunk_CE = 0

			hid_to_output_weights_gradient = hidden_layer_state.dot(error_deriv.T)
			output_bias_gradient = np.sum(error_deriv, 1)

			back_propagated_deriv_1 = hid_to_output_weights.dot(error_deriv) * hidden_layer_state * (1 - hidden_layer_state)
			embed_to_hid_weights_gradient = embedding_layer_state.dot(back_propagated_deriv_1.T)

			hid_bias_gradient = np.sum(back_propagated_deriv_1, 1)

			#150 X 100
			back_propagated_deriv_2 = embed_to_hid_weights.dot(back_propagated_deriv_1)

			global word_embedding_weights_gradient
			word_embedding_weights_gradient[:] = 0

			for w in range(0, numwords):
				#250 X 100
				expanded_input_batch = expansion_matrix[:, input_batch[w]]
				expanded_back_propagated_deriv_2 = back_propagated_deriv_2[w * numhid1: (w + 1) * numhid1, :].T
				#print expanded_input_batch.shape, back_propagated_deriv_2.shape, expanded_back_propagated_deriv_2.shape
				expanded_input_batch = expanded_input_batch.dot(expanded_back_propagated_deriv_2)
				#print expanded_input_batch.shape
				word_embedding_weights_gradient = word_embedding_weights_gradient + expanded_input_batch
			word_embedding_weights_delta = momentum * word_embedding_weights_delta \
			+ word_embedding_weights_gradient / batchsize
			word_embedding_weights = word_embedding_weights - learning_rate * word_embedding_weights_delta

			global embed_to_hid_weights_delta
			embed_to_hid_weights_delta = momentum * embed_to_hid_weights_delta \
			+ embed_to_hid_weights_gradient / batchsize
			embed_to_hid_weights = embed_to_hid_weights - learning_rate * embed_to_hid_weights_delta

			global hid_to_output_weights_delta
			hid_to_output_weights_delta = momentum * hid_to_output_weights_delta \
			+ hid_to_output_weights_gradient / batchsize
			hid_to_output_weights = hid_to_output_weights - learning_rate * hid_to_output_weights_delta

			global hid_bias_delta
			#print 'XXX', hid_bias_delta.shape, hid_bias_gradient.reshape(hid_bias_gradient.size, -1).shape
			hid_bias_delta = momentum * hid_bias_delta + hid_bias_gradient.reshape(hid_bias_gradient.size, -1) / batchsize
			hid_bias = hid_bias - learning_rate * hid_bias_delta

			global output_bias_delta
			output_bias_delta = momentum * output_bias_delta + output_bias_gradient.reshape(output_bias_gradient.size, -1) / batchsize
			output_bias = output_bias - learning_rate * output_bias_delta;

			if (m + 1) % show_validation_CE_after == 0:
				print '\rRunning validation ...',
				[embedding_layer_state, hidden_layer_state, output_layer_state] = \
				fprop(valid_input, word_embedding_weights, embed_to_hid_weights, \
					hid_to_output_weights, hid_bias, output_bias);
				datasetsize = valid_input.shape[1]
				expanded_valid_target = expansion_matrix[:, valid_target.ravel()]
				CE = -np.sum(np.sum(expanded_valid_target * np.log(output_layer_state + tiny))) /datasetsize
				print' Validation CE : ', CE
    
	print'\rAverage Training CE : ', trainset_CE

# Gather our code in a main() function
def main():
	print 'Hello World Neural Net'
	epochs = input("Enter Epochs Number: ")
	#print 'epochs', epochs
	train(epochs)

    
# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()