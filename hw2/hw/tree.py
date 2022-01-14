class Node:
	def __init__(self, label):
		self.label = label
		self.left = None
		self.right = None
		self.left_decision = None
		self.right_decision = None
		#parent = None #Try without for now
	def set_label(self, new_label):
		self.label = new_label
	def add_left(self, label):
		self.left = Node(label)
		return self.left
	def add_right(self, label):
		self.right = Node(label)
		return self.right

	def add_left(self, label):
		self.left = Node(label)
		return self.left
	def add_right(self, label):
		self.right = Node(label)
		return self.right

	def add_left_decision(self,left_dec):
		#left = yes, right = no
		#decisions are strings
		self.left_decision = left_dec
		return
	def add_right_decision(self, right_dec):
		self.right_decision = right_dec
		return
	def get_left_decision(self):
		return self.left_decision

	def get_right_decision(self):
		return self.right_decision

	def get_label(self):
		return self.label

	#note that the children of leaves are not nodes but leaves
	def get_left(self):
		return self.left
	def get_right(self):
		return self.right


