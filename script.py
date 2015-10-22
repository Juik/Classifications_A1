#hw for 2.1 

import numpy as np 
from l2_distance import l2_distance
from run_knn import run_knn
from utils import *
import matplotlib.pyplot as plt

def init():
	train_data, train_label = load_train()
	valid_data, valid_label = load_valid()
	test_data, test_label = load_test()
	
	cl_rate_valid = []
	index=[]
	for i in [1,3,5,7,9]:
		valid_labels_knn = run_knn(i, train_data, train_label, valid_data)

		num_correct_prediction = 0 
		num_total_points = 0 
		classification_rate = 0

		count = 0
		correct_count = 0
		for valid_label_knn in valid_labels_knn:
			if valid_label_knn == valid_label[count]:
				correct_count = correct_count + 1
			count = count + 1

		# print 'the classification rate of validation would be :'
		# print "%.2f%%" % (float(correct_count) / float(count)*100)

		cl_rate_valid.append(float(correct_count) / float(count))

	cl_rate_test = []
	for i in [1,3,5,7,9]:
		test_labels_knn = run_knn(i, train_data, train_label, test_data)

		count = 0
		correct_count = 0
		for test_label_knn in test_labels_knn:
			if test_label_knn == test_label[count]:
				correct_count = correct_count + 1
			count = count + 1
		# print 'the classification rate of test would be :'
		# print "%.2f%%" % (float(correct_count) / float(count)*100)
		cl_rate_test.append(float(correct_count) / float(count))
		index.append(i)

	length = len(cl_rate_test)
	for i in range(length):
		k = 2*i+1
		print 'when k = %d, classification rate of valid is %.2f, of test is %.2f' % (k, cl_rate_valid[i], cl_rate_test[i])

	plt.figure(1)
	plt.plot(index, cl_rate_test, marker='o', label='test_set')
	plt.plot(index,cl_rate_valid,marker='x',label='valid_set')
	legend = plt.legend()
	plt.grid()
	plt.xlabel('k')	
	plt.ylabel('Classification Rate')
	plt.axis([1, 9, 0.7, 1])
	plt.show()


if __name__ == '__main__':
	init()

	