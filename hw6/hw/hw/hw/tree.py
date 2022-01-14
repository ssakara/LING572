class Node:
	def __init__(self, data, parent = None):
		self.word = data[0]
		self.tag = data[1]
		self.prob = data[2]
		self.total_prob = data[3]
		self.parent = parent
		self.children = []

	def get_word(self):
		return self.word
	def get_tag(self):
		return self.tag
	def get_prob(self):
		return self.prob
	def get_total_prob(self):
		return self.total_prob
	def get_parent(self):
		return self.parent
	def add_children(self,children):
		self.children = self.children + children
		return
