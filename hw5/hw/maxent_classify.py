import sys
from decimal import Decimal
from collections import Counter, defaultdict
import warnings
import sklearn.metrics

test_data = sys.argv[1]
model_file = sys.argv[2]
output_file = sys.argv[3]

weights = defaultdict(dict)
labels = set()
with open(model_file) as model:
	label = model.readline().split()[-1]
	labels.add(label)
	for line in model:
		spl = line.split()
		if spl[0] == 'FEATURES':
			label = spl[-1]
			labels.add(label)
			continue
		weights[spl[0]][label] = Decimal(spl[-1])			

with open(test_data) as f:
	true_results = []
	results = [] 
	for line in f:
		sums = {}
		result = {}
		for label in labels:
			sums[label] = weights['<default>'][label]
		spl = line.split()
		true_results.append(spl.pop(0))
		for word in spl:
			#w_sp = word.split(':')
			for label in labels:
				sums[label] += weights[word.split(':')[0]][label]
		for label in labels:
			result[label] = sums[label].exp()			
		Z = sum(result.values())
		for label in labels:
			result[label] = result[label]/Z			
		results.append(result)

#OUTPUT FILE 

with open(output_file, 'w') as out:
		out.write('%%%%% test data:\n')
		for i, result in enumerate(results):
		#array:0 talk.politics.guns talk.politics.guns 1 talk.politics.misc 2.57623138902975e-52 talk.politics.mideast 2.32630188734125e-96
				out.write('array:' + str(i) + ' ' + true_results[i])
				label_dict = {x: result[x] for x in labels}
				for label in sorted(label_dict, key = label_dict.get, reverse = True):
						#denom = sum(result.values())
						out.write(' ' + label + ' ' + str((result[label])))
				out.write('\n')

#ACC_FILE
warnings.filterwarnings("ignore", category=FutureWarning)
#build results files
formatted_results = []
for result in results:
	formatted_results.append(max(result, key = result.get))

labels = sorted(list(labels))

print('Confusion matrix for the test data:')
print('row is the truth, column is the system output')
print()
testing_conf_mat = sklearn.metrics.confusion_matrix(true_results,formatted_results,labels)

print('\t' + ' '.join(labels))
for i, row in enumerate(testing_conf_mat):
		print(labels[i] + ' ' + ' '.join(row.astype('str')))
print()
print('Test accuracy=' + str(sklearn.metrics.accuracy_score(true_results,formatted_results)))


