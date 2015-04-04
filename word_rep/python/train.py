#!/usr/bin/env python

# import modules used here -- sys is a very standard one
import sys
import scipy.io as sio

from load import load

batchsize = 100
data = load(batchsize)
train_input = data.train_input
train_target = data.train_target
valid_input = data.valid_input
valid_target = data.valid_target
test_input = data.test_input
test_target = data.test_target
vocab = data.vocab

# Gather our code in a main() function
def main():
    print 'Hello there', train_input.shape
    # Command line args are in sys.argv[1], sys.argv[2] ...
    # sys.argv[0] is the script name itself and can be ignored

# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()