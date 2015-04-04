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
    train_target = data.trainData[D, 0:N * M].reshape(1, N, M)
    valid_input = data.validData[0:D, :]
    valid_target = data.validData[D: D + 1, :]
    test_input = data.testData[0:D, :]
    test_target = data.testData[D:D + 1, :]
    vocab = data.vocab
    R = collections.namedtuple('data', ['train_input', 'train_target', 'valid_input', 'valid_target', 'test_input', 'test_target', 'vocab'], verbose=True)
    res = R(train_input, train_target, valid_input, valid_target, test_input, test_target, vocab)
    return res
