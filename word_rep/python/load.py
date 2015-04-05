#!/usr/bin/env python

# import modules used here -- sys is a very standard one
import sys
import math
import collections
import scipy.io as sio

def load(N):
    matfile = sio.loadmat('../data/data.mat', squeeze_me=True, struct_as_record=False)
    data = matfile['data']
    numdims = data.trainData.shape[0]
    D = numdims - 1
    M = math.floor(data.trainData.shape[1]/N)
    
    train_input = data.trainData[0:D, 0:N * M].reshape(D, N, M)
    train_input = train_input - 1 #the start idx in python is 0

    train_target = data.trainData[D, 0:N * M].reshape(1, N, M)
    train_target = train_target - 1

    valid_input = data.validData[0:D, :]
    valid_input = valid_input - 1

    valid_target = data.validData[D: D + 1, :]
    valid_target = valid_target - 1 

    test_input = data.testData[0:D, :]
    test_input = test_input - 1

    test_target = data.testData[D:D + 1, :]
    test_target = test_target - 1
    
    vocab = data.vocab
    #R = collections.namedtuple('data', ['train_input', 'train_target', 'valid_input', 'valid_target', 'test_input', 'test_target', 'vocab'], verbose=True)
    #res = R(train_input, train_target, valid_input, valid_target, test_input, test_target, vocab)
    #return res
    return (train_input, train_target, valid_input, valid_target, test_input, test_target, vocab)
    
