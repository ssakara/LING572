import sys
from collections import Counter, defaultdict
from math import log
from tree import Node


#Read in file
def data_in(training_data):
	train = []
	label_count = Counter()
	features = set()
	#train
	with open(training_data) as f:
		f.seek(0,0)
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



def split(data, label_count, features, min_gain, cur_depth=-1, node = None, feat_path = [], direction = None, root = None): 
	#this will be the first split but then I think we can generalize it so we can call recursively
	
	#add left and right children that lead to biggest info gain
	#node is the parent node, in this code we're creating it's left or right child with the right label
	#first determine which feature to split on

	min_ent = cond_ent(data, label_count, features, feat_path)
	infoGain = entropy(label_count) - min_ent[0]
	feat = min_ent[1]
	print(feat)
	if max_depth == cur_depth or infoGain < min_gain:
	#leaf handling
		try: root
		except NameError:
			return
		if direction == 'left':
			node.add_left_decision(label_count.most_common(1)[0][0])		
		if direction == 'right':
			node.add_right_decision(label_count.most_common(1)[0][0])		
		return root
	if node == None:
		root = new_node = Node(feat)
	elif direction == 'left':
		new_node = node.add_left(feat)
	elif direction == 'right':
		new_node.add_right(feat)
	feat_path.append(feat)

	#now actually split the data - left = yes, right = no
	#I don't think we need to retain order - if we do we should use a deque
	data_left = []
	label_count_left = Counter()
	features_left = set()
	data_right = []
	label_count_right = Counter()
	features_right = set()
	while len(data) > 0:
		line = data.pop()
		if feat in line:
			data_left.append(line)
			label_count_left[line[1]] += 1
			features_left.add(feat)
		else:
			data_right.append(line)
			label_count_right[line[1]] += 1
			features_right.add(feat)
	if len(features_left)==0:
		print('left is empty')
	else:
		print('left is nonempty')		
	if len(features_right)==0:
		print('right is empty')
	else:
		print('right is nonempty')		
	cur_depth += 1
	#call yourself recursively - the first line isn't altering the second line's parameters, is it??
	split(data_left, label_count_left, features_left, min_gain, cur_depth, new_node, feat_path, 'left', root)
	split(data_right, label_count_right, features_right, min_gain, cur_depth, new_node, feat_path, 'right', root)

def test_vector(vector, node):
	#vector is [set(features), label]
	if node.get_label in vector[0]: #yes
		if node.get_left():
			test(vector, node.get_left())
		else: #we've reached a leaf
			return node.get_left_decision()
	else: #no
		if node.get_right():
			test(vector, node.get_right())

		else: #we've reached a leaf
			return node.get_right_decision()

def test(data_file, node):
	test, label_count, features = data_in(data_file)
	results = []
	for line in test:
		results.append(test_vector(line,node)) 
	return results
#def infoGain(data, feat, label_count):
#	return entropy(label_count) - cond_ent(data, feat, label_count)[0]

def entropy(label_count):
	#entropy doesn't care about features, only labels
	#I don't think we need to even read in data

	sum_all = sum(label_count.values())
	ans = 0
	for label in label_count:
		ans += -log(label_count[label]/sum_all,2)*(label_count[label]/sum_all)

	return ans

def print_tree(node):
	node = node.get_left()
	print(node)
	print_tree(node)
	node = node.get_right()
	print(node)
	print_tree(node)

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
			sum_feat = sum(feat_count[feat].values()) 
			sum_not_feat = sum(not_feat_count[feat].values())
			sum_all = sum_feat + sum_not_feat

			ans_feat = 0
			for label in feat_count:
				if feat_count[feat][label] == 0:
					ans_feat += 0
				else:
					ans_feat += (sum_feat/sum_all)*(-log(feat_count[feat][label]/sum_feat,2)*(feat_count[feat][label]/sum_feat))

			ans_not_feat = 0
			for label in not_feat_count:
				if feat_count[feat][label] == 0:
					ans_not_feat += 0
				else:
					ans_not_feat += (sum_not_feat/sum_all)*(-log(not_feat_count[feat][label]/sum_not_feat,2)*(not_feat_count[feat][label]/sum_not_feat))
			
			ans = ans_feat + ans_not_feat

			if min_ent == []:
				min_ent = [ans, feat]
			else:
				if ans < min_ent[0]:
					min_ent = [ans, feat]

	return min_ent

	

#training_data = sys.argv[1]
#test_data = sys.argv[2]
#max_depth = int(sys.argv[3])
#min_gain = float(sys.argv[4])
#model_file = sys.argv[5]
#sys_output = sys.argv[6]

#build dt.sh training_data test_data max_depth min_gain model_file sys_output > acc_file
training_data = 'train.vectors.txt' 
test_data = 'test.vectors.txt'
max_depth = 2
min_gain = .1

data, label_count, features = data_in(training_data)
#root = split(data, label_count, features, min_gain)
#print(root)
#print_tree(root) 
#results = test(test_data, root)
#print(results)
