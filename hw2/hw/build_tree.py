import sys
from collections import Counter, defaultdict
from math import log
from tree import Node
from decimal import Decimal
import build_output_files

#Read in file
def data_in(training_data):
	train = []
	label_count = Counter()
	features = set()
	#train
	with open(training_data) as f:
		for line in f:
			spl = line.split()
			label = spl.pop(0)
			feats = set()
			#print(spl)
			for pair in spl:
				sp = pair.split(':')
				feat = pair.split(':')[0]
				feats.add(feat)
				features.add(feat)
			train.append((feats, label))
			label_count[label] += 1 #not sure if I want this here
	#print(features)
	return train, label_count, features

#build tree

def get_paths_master(root):
	path_list = get_paths(root, [], [])
	return path_list

def get_paths(node, path, master_path_list):
	#store in a list of lists
	#print(node.get_label())
	path_left = path + [node.get_label()]
	path_right = path + ['!' + node.get_label()]

	if node.get_left():
		left_path_list = get_paths(node.get_left(), path_left, master_path_list)
		right_path_list = get_paths(node.get_right(), path_right, master_path_list)
		#master_path_list = left_path_list + right_path_list
		oaster_path_list = right_path_list + left_path_list
		return master_path_list
	else:
		master_path_list.append(path_left)
		master_path_list.append(path_right)
		return master_path_list


def paths_list_to_dict(paths_list):
	#for path in lists:
	return
	

def train_traverse(vector, node, path):
	#vector [{features}, label]
	label = node.get_label()
	if label in vector[0]:
		path.append(label)
		if node.get_left():
			path, result = train_traverse(vector, node.get_left(), path)
		else:
			result = node.get_left_decision()
			return path, result
	else:
		path.append('!' + label)
		if node.get_right():
			path, result = train_traverse(vector, node.get_right(), path)
		else:
			result = node.get_right_decision()
			return path, result
	return path, result
def model_results(paths_list):
	#paths_list is a list of lists

	return

def split(data, label_count, features, min_gain, cur_depth, max_depth, node, feat_path, labels):
	#the first time use node = Node(''):, cur_depth = 0, feat_path = []
	min_ent = cond_ent(data, label_count, features, feat_path)
	infoGain = entropy(label_count) - min_ent[0]
	feat = min_ent[1]
	node.set_label(feat)
	#print(feat)
	feat_path.append(feat)
	#split the data
	data_left = []
	label_count_left = Counter()
	features_left = set()
	data_right = []
	label_count_right = Counter()
	features_right = set()
	while len(data) > 0:
		line = data.pop()
		if feat in line[0]:
			data_left.append(line)
			label_count_left[line[1]] += 1
			for feature in line[0]:
				features_left.add(feature)
		else:
			data_right.append(line)
			label_count_right[line[1]] += 1
			for feature in line[0]:
				features_right.add(feature)
	#if we need to make this a leaf
	
	#print([max_depth, cur_depth, infoGain])
	#print(len(label_count_left))
	if max_depth == cur_depth or infoGain < min_gain or len(features_left - set(feat_path)) == 0 or len(features_right - set(feat_path)) == 0:
		if len(label_count_left) != 0:
			node.add_left_decision(label_count_left.most_common(1)[0][0])
		else:
			node.add_left_decision(labels[0])
		if len(label_count_right) != 0:
			node.add_right_decision(label_count_left.most_common(1)[0][0])
		else:
			node.add_right_decision(labels[-1])
		return 
	cur_depth += 1

	#create the new nodes
	node_left = node.add_left('')
	node_right = node.add_right('')
	#now split the left and right children
	_ = split(data_left, label_count_left, features_left, min_gain, cur_depth, max_depth, node_left, feat_path, labels)
	_ = split(data_right, label_count_right, features_right, min_gain, cur_depth, max_depth, node_right, feat_path, labels)
	return node


def test_vector(vector, node):
	
	#vector is [set(features), label]
	if node.get_label in vector[0]: #yes
		if node.get_left():
			test_vector(vector, node.get_left())
		else: #we've reached a leaf
			return node.get_left_decision()
	else: #no
		if node.get_right():
			test_vector(vector, node.get_right())

		else: #we've reached a leaf
			return node.get_right_decision()

def test(test_data_file, node):
	#test_data_file is output of data_in for test_data
	#results is a list of the results for each test vector
	#test, label_count, features = data_in(data_file)
	results = []
	for line in test_data_file:
		results.append(test_vector(line,node)) 
	return results

def entropy(label_count):
	#entropy doesn't care about features, only labels
	#I don't think we need to even read in data

	sum_all = Decimal(sum(label_count.values()))
	ans = Decimal(0)
	for label in label_count:
		ans += Decimal(-log(label_count[label]/sum_all,2))*(Decimal(label_count[label]/sum_all))

	return ans


def cond_ent(data, label_count, features, feat_path):
	#returns minimum cond_ent and the feature that leads to it
	##What we're now trying to do is calculate cond ent for all features and then find the feature that leads to the minimum!!
	#ignore features in feat_path they're aolready usted (not sure this is necessary, not doing it yet)
	feat_count = defaultdict(Counter) #dict feat: (label: count)
	not_feat_count = defaultdict(Counter) #dict feat: (label: count)
	min_ent = []

	#features = set()
	#for line in data:
	#	for word in line[0]:
	#		features.add(word)
	#print(features)
	#if feat_path == ['israel']:
	#	print(features)
	for feat in features: #can we avoid this?
		if feat not in feat_path:
			for line in data:
				if feat in line[0]:
					feat_count[feat][line[1]] += 1
				else:
					#print(type(line))
					#print(line[0])
					#print(line[1])
					not_feat_count[feat][line[1]] += 1
			#now calc cond ent and compare to current min. call the cond ent 'min' and account for it not existing = see other screen for how to do this!
			sum_feat = Decimal(sum(feat_count[feat].values())) 
			sum_not_feat = Decimal(sum(not_feat_count[feat].values()))
			sum_all = Decimal(sum_feat + sum_not_feat)

			ans_feat = Decimal('0')
			for label in feat_count[feat]:
				if feat_count[feat][label] == 0:
					ans_feat += 0
				else:
					ans_feat += (sum_feat/sum_all)*Decimal(-log(feat_count[feat][label]/sum_feat,2))*(feat_count[feat][label]/sum_feat)

			ans_not_feat = Decimal('0')
			for label in not_feat_count[feat]:
				if not_feat_count[feat][label] == 0:
					ans_not_feat += 0
				else:
					ans_not_feat += (sum_not_feat/sum_all)*Decimal(-log(not_feat_count[feat][label]/sum_not_feat,2))*(not_feat_count[feat][label]/sum_not_feat)

			ans = ans_feat + ans_not_feat
			#if feat_path == ['israel']:
			#	print([ans, feat])
			if min_ent == []:
				min_ent = [ans, feat]
			else:
				if ans < min_ent[0]:
					min_ent = [ans, feat]
	#print(min_ent)
	#if min_ent ==  []:
	#	print(features)
	#	min_ent=[Decimal('0'), list(features)[0]]	
	return min_ent

	
#MAIN
training_data = sys.argv[1]
#print(training_data)
test_data = sys.argv[2]
max_depth = int(sys.argv[3])
min_gain = Decimal(sys.argv[4])
model_file = sys.argv[5]
sys_output = sys.argv[6]

#build dt.sh training_data test_data max_depth min_gain model_file sys_output > acc_file
#training_data = 'train.vectors.txt' 
#test_data = 'test.vectors.txt'
#max_depth = 2
#min_gain = Decimal(.1)
#model_file = 'model1'
#sys_output = 'sys1'

#TRAINING (building the tree)
data, label_count, features = data_in(training_data)
labels = sorted(list(label_count.keys()))
#print(data)
root = split(data, label_count, features, min_gain, 0, max_depth, Node(''), [], labels)
all_paths = get_paths(root, [], []) 

#TRAVERSING WITH TRAINING DATA
data, _, _ = data_in(training_data)
path_to_training_results = defaultdict(Counter)
training_results = []
training_vector_paths = []
true_training_results = []
for vector in data:
	path, result = train_traverse(vector,root,[])
	path_to_training_results[tuple(path)][vector[1]] += 1
	training_vector_paths.append(path)
	training_results.append(result)
	true_training_results.append(vector[1])
#print(training_results)
#print(true_training_results)
#TESTING
test_data_vectors, _, _ = data_in(test_data)
#testing_results = test(test_data_vectors, root)
path_to_testing_results = defaultdict(Counter)
testing_results = []
testing_vector_paths = []
true_testing_results = []
for vector in test_data_vectors:
	path, result = train_traverse(vector,root,[])
	path_to_testing_results[tuple(path)][vector[1]] += 1
	testing_vector_paths.append(path)
	testing_results.append(result)
	true_testing_results.append(vector[1])
	
build_output_files.build_model_file(all_paths, path_to_training_results, labels, model_file)
build_output_files.build_sys_output(training_vector_paths,path_to_training_results, testing_vector_paths, path_to_testing_results, labels, sys_output) #NEED TO HANDLE TESTING TOO!!
build_output_files.build_acc_file(true_training_results, training_results, true_testing_results, testing_results, labels)
#print(paths_list)
#print(results)
