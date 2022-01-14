from collections import Counter, defaultdict
import sklearn.metrics

def build_model_file(all_paths, path_to_result, labels, model_file):
	#path_to_result = path: (label: #)
	with open(model_file, 'w+') as out:
		for path in all_paths:
			tot = sum(path_to_result[tuple(path)].values())
			out.write('&'.join(path) + ' ')
			out.write(str(tot)) 
			if tot == 0:
				for label in sorted(labels):
					out.write(' ' + label + ' ' + '0')
			else:
				for label in sorted(labels):
					out.write(' ' + label + ' ' + str(path_to_result[tuple(path)][label]/tot))
			out.write('\n')

def build_sys_output(training_vector_paths,path_to_training_results, testing_vector_paths, path_to_testing_results, labels, sys_output):
	with open(sys_output, 'w+') as out:
		out.write('%%%%% training data:\n')
		#TRAINING
		for i, path in enumerate(training_vector_paths):
			tot = sum(path_to_training_results[tuple(path)].values())
			out.write('array:' + str(i) + ' ')
			line = []
			for label in sorted(labels):
				line.append(label)
				line.append(str(path_to_training_results[tuple(path)][label]/tot))
			out.write('\t'.join(line) + '\n')
				#out.write(label + '\t' + str(path_to_result[tuple(path)][label]/tot)

		out.write('\n\n')	
		#TESTING
		out.write('%%%%% test data:\n')
		for i, path in enumerate(testing_vector_paths):
			tot = sum(path_to_testing_results[tuple(path)].values())
			out.write('array:' + str(i) + ' ')
			line = []
			for label in sorted(labels):
				line.append(label)
				line.append(str(path_to_testing_results[tuple(path)][label]/tot))
			out.write('\t'.join(line) + '\n')

def build_acc_file(true_training_results, training_results, true_testing_results, testing_results, labels):
	labels = sorted(labels)
	print('Confusion matrix for the training data:')
	print('row is the truth, column is the system output')
	print()
	training_conf_mat = sklearn.metrics.confusion_matrix(true_training_results,training_results,labels)
	#training_acc = sklearn.metrics.accuracy_score(true_training_results,training_results)

	print('\t' + ' '.join(labels))
	for i, row in enumerate(training_conf_mat):
		print(labels[i] + ' ' + ' '.join(row.astype('str')))
	print()
	print('Training accuracy=' + str(sklearn.metrics.accuracy_score(true_training_results,training_results)))
	print()
	print()
	print('Confusion matrix for the test data:')
	print('row is the truth, column is the system output')
	print()
	testing_conf_mat = sklearn.metrics.confusion_matrix(true_testing_results,testing_results,labels)

	print('\t' + ' '.join(labels))
	for i, row in enumerate(testing_conf_mat):
		print(labels[i] + ' ' + ' '.join(row.astype('str')))
	print()
	print('Test accuracy=' + str(sklearn.metrics.accuracy_score(true_testing_results,testing_results)))
