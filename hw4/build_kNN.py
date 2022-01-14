import sys
from collections import Counter, defaultdict
from decimal import Decimal
from math import log
import sklearn.metrics
import sklearn.feature_extraction 
import warnings
import numpy as np
import outputs

training_data = sys.argv[1]
test_data = sys.argv[2]
k = int(sys.argv[3])
similarity_func = int(sys.argv[4]) #1=euc, 2=cos
sys_output = sys.argv[5]

#training_data = 'train.vectors.txt'
#test_data = 'test.vectors.txt'
#k = 5 
#similarity_func = 2 
#sys_output = 'sys.out1'

true_training_results = []
train_vec_list = []
true_testing_results = []
test_vec_list = []
v = sklearn.feature_extraction.DictVectorizer(sparse=True)
labels = set()
with open(training_data) as f:
	for line in f:
		spl = line.split()
		label = spl.pop(0)
		labels.add(label)
		true_training_results.append(label)
		vec = {word.split(':')[0]:int(word.split(':')[1]) for word in spl}
		train_vec_list.append(vec)
train = v.fit_transform(train_vec_list)

with open(test_data) as f:
	for line in f:
		spl = line.split()
		true_testing_results.append(spl.pop(0))
		vec = {word.split(':')[0]:int(word.split(':')[1]) for word in spl}
		test_vec_list.append(vec)
test = v.transform(test_vec_list)

#training_results = []
if similarity_func == 1:
	#distances = sklearn.metrics.pairwise.euclidean_distances(train, test)
	train_inds = np.argpartition(sklearn.metrics.pairwise.euclidean_distances(train, train), k)[:,:k]
	test_inds = np.argpartition(sklearn.metrics.pairwise.euclidean_distances(test, train), k)[:,:k]
	#training_results = [Counter(x).most_common()[0][0] for x in inds]

elif similarity_func ==2:
	train_inds = np.argpartition(sklearn.metrics.pairwise.cosine_similarity(train, train), -k)[:,-k:]
	test_inds = np.argpartition(sklearn.metrics.pairwise.cosine_similarity(test, train), -k)[:,-k:]
training_results = [Counter(np.array(true_training_results)[x]) for x in train_inds]
testing_results = [Counter(np.array(true_training_results)[x]) for x in test_inds]

outputs.output_sys_file(testing_results, true_testing_results, training_results, true_training_results, labels, sys_output)		 
outputs.output_acc_file(testing_results, true_testing_results, training_results, true_training_results, labels)	






