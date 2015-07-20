#!/usr/bin/env python

# import modules used here -- sys is a very standard one
import sys, getopt
import collections
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
	if number[idx + 1] != '-' and number[idx + 1] != '+':
		print number
		raise NameError("Error: process efloat")
	return int(number[idx + 2:])



def isfloat(value):	
	
	try:
		return (float(value) < 1)
	except ValueError:
		return False


def parse_matrix_info(line):
	if not "epoch" in line or not "batch" in line:
		print line
		raise NameError("Error parse matrix info")
	infos = line.split()
	# print len(infos)
	# epoch=''
	# batch=''
	# row=''
	# column=''
	for info in infos:
		idx = info.index(':')
		# print info
		if "epoch" in info:
			epoch = int(info[idx + 1:])
		elif "batch" in info:
			batch = int(info[idx + 1:])
		elif "row" in info:
			row = int(info[idx + 1:])
		elif "column" in info:
			column = int(info[idx + 1:])
		else:
			print line
			raise NameError("Error parse matrix info")

	return (epoch, batch, row, column)
 

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

	
	initial = False
	precises = {}
	epoch_precises = {}
	old_epoch = 0

	with open(inputfile, "r") as infile:
		for line in infile:
			
			if "epoch" in line:
				(epoch, batch, row, column) = parse_matrix_info(line)
				initial = True
				# print "epoch %d batch %d row %d column %d " %(epoch, batch, row, column)
				# print precises
				precises.clear()

				if old_epoch != epoch:
					print epoch 
					print epoch_precises
					epoch_precises.clear()
					old_epoch = epoch

			elif initial:
				numbers = line.split()

				for num in numbers:
					if isfloat(num):
						p = -1
						if 'e' in num:
							p = process_efloat(num)	
						else:
							p = process_float(num)

						if p == -1:
							continue

						if p not in precises:
							precises[p] = 1
						else:
							precises[p] = precises[p] + 1

						if p not in epoch_precises:
							epoch_precises[p] = 1
						else:
							epoch_precises[p] = epoch_precises[p] + 1

	print epoch 
	print epoch_precises

# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
	main(sys.argv[1:])