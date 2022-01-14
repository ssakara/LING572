import sys
import tree
from collections import Counter, defaultdict, deque
from decimal import Decimal
import heapq
import sklearn.metrics
import time

#read in boundary file and write to a data struct
#read in model file and write to a data struct

def model_file_in(model_file):
	labels = set()
	weights = defaultdict(lambda: defaultdict(int))
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
	return labels, weights

def boundary_file_in(boundary_file):
	with open(boundary_file) as f:
		boundaries = [int(line.strip()) for line in f]
	return boundaries

def base_weight_sum(features, tags, weights):
	sums = {}
	for tag in tags:
		sums[tag] = weights['<default>'][tag]
	for feat in features:
		for tag in tags:
			sums[tag] += weights[feat][tag]
	return sums

def decode_pt2(sums, tags, new_features, topN):
	result = {}
	for feat in new_features:
		for tag in tags:
			sums[tag] += weights[feat][tag]	
	for tag in tags:
		result[tag] = sums[tag].exp()			
	Z = sum(result.values())
	result = dict(Counter(result).most_common(topN)) #NEW
	for tag in result:
		result[tag] = result[tag]/Z			
	#results.append(result)
	return result

def maxent_decode(features, tags, weights, topN, summ = None):
	#features: each feature for word that we're classifying
	#labels: set of tags
	#return: P(y|x) for each label y. return it as a dict label:prob
	sums = {}
	result = {}
	for tag in tags:
		sums[tag] = weights['<default>'][tag]
	#spl = line.split()
	for feat in features:
		#w_sp = word.split(':')
		for tag in tags:
			sums[tag] += weights[feat][tag]
	for tag in tags:
		result[tag] = sums[tag].exp()			
	Z = sum(result.values())
	result = dict(Counter(result).most_common(topN)) #NEW
	for tag in result:
		result[tag] = result[tag]/Z			
	#results.append(result)
	return result

def output_sys_files(results, true_results, sys_output): #IT'S NOT PRINTING THIS ALL OF THE SUDDEN SOMETHING IS BROKEN
	with open(sys_output, 'w+') as out:
		for i, result in enumerate(results):
			for j, line in enumerate(result): #results is reversed but true_results is not #NOT ANYMORE
				out.write(str(i + 1) + '-' + str(j) + '-' + line[0] + ' ' + str(true_results[i][j]) + ' ' + line[1] + ' ' + str(line[2]) + '\n')
	return

def output_test_acc(results, true_results):
	formatted_results = [r[1] for result in results for r in result]
	formatted_true_results = [t for true_result in true_results for t in true_result]
	print('Test accuracy=' + str(sklearn.metrics.accuracy_score(formatted_true_results,formatted_results)))
	return

test_file = sys.argv[1]
boundary_file = sys.argv[2]
model_file = sys.argv[3]
sys_output = sys.argv[4]
beam_size = int(sys.argv[5])	
topN = int(sys.argv[6])	
topK = int(sys.argv[7])	

#model_file = 'm1.txt'
#boundary_file = 'boundary.txt'
#test_file = 'test.txt'
#beam_size = 2
#topN = 5
#topK = 10
#sys_output = 'out1'

tags, weights = model_file_in(model_file)
boundaries = boundary_file_in(boundary_file)
results = [] #lists of lists where each list is a tuple (word, tag, prob)
true_results = []
nodes = []
true_results_line = []
#result_num = 0

with open(test_file) as f:
	t0 = time.time() #start timing
	for line in f:
		features = set()
		spl = line.split('-',2)
		line_num = int(spl[0])
		word_num = int(spl[1]) 
		data_spl = spl[2].split()
		word = data_spl.pop(0)
		true_results_line.append(data_spl.pop(0))

		features = set(data_spl[::2])

		#use feat_counts to determine top N tags, with adding the features you need based on the current path
		if word_num == 0:
			features.add('prevT=BOS')
			features.add('prevTwoTags=BOS+BOS')	
			result = maxent_decode(features, tags, weights, topN) #decode function	
			nodes.append([tree.Node((word, tag, result[tag], result[tag])) for tag in result])
		else:
			sums = base_weight_sum(features, tags, weights)
			this_level = []
			min_node = tree.Node((None,None,2,2))
			for node in nodes[word_num-1]: 
				new_features = set()
				if word_num == 1:
					new_features.add('prevT=' + node.get_tag())
					new_features.add('prevTwoTags=BOS+' + node.get_tag())
				else:
					new_features.add('prevT=' + node.get_tag())
					new_features.add('prevTwoTags=' + node.get_parent().get_tag() + '+' + node.get_tag())	
				result = decode_pt2(sums, tags, new_features, topN) #put the topN stuff into the result function

				for tag in result:
					if len(this_level) < topK:
						to_add = tree.Node((word, tag, result[tag], result[tag]*node.get_total_prob()), node)
						this_level.append(to_add)
						if result[tag]*node.get_total_prob() < min_node.get_total_prob():
							min_node = to_add 
						#else:
							#print(result[tag]*node.get_total_prob())
							#min_node = min(this_level, key=lambda x:x.get_total_prob())
					else:
						if result[tag]*node.get_total_prob() > min_node.get_total_prob():
							#print(min_node)
							this_level.remove(min_node)
							this_level.append(tree.Node((word, tag, result[tag], result[tag]*node.get_total_prob()), node))
							min_node = min(this_level, key=lambda x:x.get_total_prob())
						#else:
							#min_node = min(this_level, key=lambda x:x.get_total_prob())
			nodes.append(this_level)	

		if word_num == boundaries[line_num-1] - 1: #reached end of instance
			node = max(nodes[-1], key=lambda x:x.get_total_prob())
			path = deque() #list of tuples word, tag, prob
			while True:
				path.appendleft((node.get_word(), node.get_tag(), node.get_prob()))
				if not node.get_parent():
					break
				node = node.get_parent()
			results.append(path)
			#result_num += 1
			#if result_num % 100 == 0:
			#	sys.stderr.write(str(result_num))
			nodes = []
			true_results.append(true_results_line)
			true_results_line = []
	sys.stderr.write(str(time.time() - t0) + '\n') #finish timing

output_sys_files(results, true_results, sys_output)
output_test_acc(results, true_results)
