import sys
import math
import collections
import scipy.io as sio
import numpy as np


def fprop(input_batch, word_embedding_weights, embed_to_hid_weights, \
			hid_to_output_weights, hid_bias, output_bias):

	[numwords, batchsize] = input_batch.shape
	[vocab_size, numhid1] = word_embedding_weights.shape
	numhid2 = embed_to_hid_weights.shape[1]

	#embedding_layer_idx = input_batch.flatten()
	embedding_layer_idx = input_batch.T.ravel()

	#150 X 100
	embedding_layer_state = \
	word_embedding_weights[embedding_layer_idx].reshape(batchsize, -1).T

	inputs_to_hidden_units = embed_to_hid_weights.T.dot(embedding_layer_state) + hid_bias

	hidden_layer_state = 1 / (1 + np.exp(-inputs_to_hidden_units))

	inputs_to_softmax = hid_to_output_weights.T.dot(hidden_layer_state) + output_bias

	inputs_to_softmax = inputs_to_softmax - np.amax(inputs_to_softmax)

	output_layer_state = np.exp(inputs_to_softmax)

	output_layer_state = output_layer_state/np.sum(output_layer_state)

	return (embedding_layer_state, hidden_layer_state, output_layer_state)