import sys
from collections import defaultdict, Counter

training_data = sys.argv[1]
output_file = sys.argv[2] 

counts = defaultdict(Counter)
labels = set()
N = 0
with open(training_data) as f:
	for line in f:
		spl = line.split()
		label = spl.pop(0)
		labels.add(label)
		for word in spl:
			counts[word.split(':')[0]][label] += 1
		N += 1

with open(output_file, 'w+') as out:
	for label in sorted(labels):
		for word in sorted(counts.keys()):
			out.write(label + ' ' + word + ' ' + str(counts[word][label]/N)+ ' ' + str(counts[word][label]) + '\n')	
	
