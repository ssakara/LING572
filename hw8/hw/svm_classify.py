import sys
import numpy as np
import sklearn.feature_extraction
import sklearn.metrics
import scipy.sparse

test_data = sys.argv[1]
model_file = sys.argv[2]
output_file = sys.argv[3]

#test_data = 'test'
#model_file = 'model.1'
#output_file = 'out1'
#

def files_in(model_file, test_file):
	weights = []
	true_testing_results = []
	sv_list = []
	test_vec_list = []
	gamma = coef0 = degree = None
	v = sklearn.feature_extraction.DictVectorizer(sparse=True)
	with open(model_file) as f:
		f.readline()
		kernel_type = f.readline().split()[1]
		for line in f:
			#also need to get gamma, coef0, and degree values if they exist
			if line.split()[0] == 'rho':
				rho = float(line.split()[1])
			if line.split()[0] == 'gamma':
				gamma = float(line.split()[1])
			if line.split()[0] == 'coef0':
				coef0 = float(line.split()[1])
			if line.split()[0] == 'degree':
				degree = float(line.split()[1])
			elif line.strip() == 'SV':
				break
		for line in f:
			spl = line.split()
			weights.append(float(spl.pop(0)))
			#true_training_results.append(label)
			vec = {word.split(':')[0]:int(word.split(':')[1]) for word in spl}
			sv_list.append(vec)
	sv = v.fit_transform(sv_list)

	with open(test_data) as f:
		for line in f:
			spl = line.split()
			true_testing_results.append(int(spl.pop(0)))
			vec = {word.split(':')[0]:int(word.split(':')[1]) for word in spl}
			test_vec_list.append(vec)
	test = v.transform(test_vec_list)
	return sv, test, weights, rho, true_testing_results, kernel_type, gamma, coef0, degree

def decode(sv, test, weights, rho, kernel_type, gamma = None, coef0 = None, degree = None):
	if kernel_type == 'linear':
		results = test.dot(sv.T) * np.array(weights).T - rho #need to account for all kernals though...
	if kernel_type == 'polynomial':
		#results = (gamma * test.dot(sv.T) + coef0).power(degree) * np.array(weights).T - rho
		results = scipy.sparse.csc_matrix(gamma * test.dot(sv.T) + coef0*np.ones((gamma * test.dot(sv.T)).shape)).power(degree) * np.array(weights).T - rho
	if kernel_type == 'rbf':
		results = np.array([np.dot(np.exp(-gamma * sklearn.metrics.pairwise_distances(sv,row).flatten() ** 2).T, np.array(weights).T) - rho for row in test])
	#SIGMOID - last thing to do, then work on outputs
	#add gamma/coef0/degree handling too
	if kernel_type == 'sigmoid':
		results = scipy.sparse.csc_matrix(np.tanh(gamma * test.dot(sv.T) + coef0*np.ones((gamma * test.dot(sv.T)).shape))) * np.array(weights).T - rho
		#results = gamma * test.dot(sv.T)
		#results.data += coef0
		#results = np.tanh(results) * np.array(weights).T - rho
	return results

def sign_map(num):
	if num < 0:
		return 1
	else:
		return 0

def output_sys_file(results, true_testing_results, output_file):
	formatted_results = np.vectorize(sign_map)(results)
	with open(output_file, 'w+') as out:
		for i,result in enumerate(true_testing_results):
			out.write(str(result) + ' ' + str(formatted_results[i]) + ' ' + str(results[i]) + '\n')		
	print('Test accuracy=' + str(sklearn.metrics.accuracy_score(true_testing_results,formatted_results)))

sv, test, weights, rho, true_testing_results, kernel_type, gamma, coef0, degree = files_in(model_file, test_data)
results = decode(sv, test, weights, rho, kernel_type, gamma, coef0, degree)
#formatted_results = np.vectorize(svm_classify.sign_map)(results)
output_sys_file(results, true_testing_results, output_file)
