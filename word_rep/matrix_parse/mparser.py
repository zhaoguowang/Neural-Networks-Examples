#!/usr/bin/env python

# import modules used here -- sys is a very standard one
import sys, getopt
import os
import math
import numpy as np
import scipy.io as sio

inputfile = ''
outputfile = ''

def process_float(number):
	precision = 0
	for p in number:
		if p == '-' or p == '.':
			continue;
		elif p == '0':
			precision = precision + 1
		else:
			return precision

def process_efloat(number):
	idx = number.index('e')
	if number[idx + 1] != '-':
		raise NameError('Error: process efloat')
	return number[idx + 2:]



def isfloat(value):	
	
	try:
		return (float(value) < 1)
	except ValueError:
		return False

# Gather our code in a main() function
def main(argv):

	print 'Hello World Matrix Parser'
	try:
		opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   	except getopt.GetoptError:
   		print 'mparser.py -i <inputfile> -o <outputfile>'
   		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'mparser.py -i <inputfile> -o <outputfile>'
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg
		elif opt in ("-o", "--ofile"):
			outputfile = arg
	print 'Input file is "', inputfile
	print 'Output file is "', outputfile

	with open(inputfile, "r") as infile:
		for line in infile:
			numbers = line.split()
			for num in numbers:
				if isfloat(num):
					if 'e' in num:
						print process_efloat(num)
					else:
						 process_float(num)



# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
	main(sys.argv[1:])