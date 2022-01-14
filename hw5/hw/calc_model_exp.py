import sys
from collections import Counter, defaultdict
from decimal import Decimal

weights = defaultdict(dict)
counts = defaultdict(Counter)
labels = set()
#results = []
#sumss = []
training_data = sys.argv[1]
output_file = sys.argv[2]

if len(sys.argv) == 4:
	model_file = sys.argv[3] 
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

N = 0
with open(training_data) as f:
	if len(sys.argv) == 4:
		for line in f:
			sums = {}
			result = {}
			spl = line.split()
			del spl[0]
			#labels.add(spl.pop(0))
			for label in labels:
				sums[label] = weights['<default>'][label]
			for word in spl:
				for label in labels:
					sums[label] += weights[word.split(':')[0]][label]
			for label in labels:
				result[label] = sums[label].exp()			
			Z = sum(result.values())
			for label in labels:
				result[label] = result[label]/Z			
			#calc model exp
			for word in spl:
				for label in labels:
					counts[word.split(':')[0]][label] += result[label]
			N += 1
			#results.append(result)
			#sumss.append(sums)
	else:
		for line in f:
			spl = line.split()
			labels.add(spl.pop(0))
		C = len(labels)
		f.seek(0,0)
		for line in f:
			spl = line.split()
			del spl[0]
			for word in spl:
				for label in labels:
					counts[word.split(':')[0]][label] += 1/C
			N += 1

with open(output_file, 'w+') as out:
	for label in sorted(labels):
		for word in sorted(counts.keys()):
			out.write(label + ' ' + word + ' ' + str(counts[word][label]/N)+ ' ' + str(counts[word][label]) + '\n')	
