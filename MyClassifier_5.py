import numpy as np

class ifier:
	def __init__(self, class1, class2, num_features, alpha=1, lamda=1, epochs=100):
		self.c=np.array([class1,class2])
		self.M=num_features
		self.train_set=np.empty((0,num_features))
		self.w=np.zeros(self.M)
		self.b=0
		self.alpha=alpha
		self.lamda=lamda
		self.epochs=epochs

	def sample_selection(self, training_sample):
		'''
		Input:
		training_sample		:	numpy array of shape (M,1)	:	current sample

		Output:
		self.train_set[]	:	numpy array of shape (N,M)	:	append current_sample to this if selected
		is_selected			:	bool						:	whether current sample selected or not
		'''

		# code here for checking whether curr_sample should be included
		is_selected=1

		if is_selected:
			self.train_set=np.append(self.train_set, np.reshape(training_sample,(1,len(training_sample))), axis=0)
		
		return is_selected

	def train(self, train_data, train_label):
		'''
		Input:
		train_data	:	numpy array of shape (N,M)	:	features
		train_label	:	numpy array of shape (N,)	:	labels
		self.alpha	:	float						:	learning rate
		self.lamda	:	float						:	regularization param
		self.epochs	:	int							:	number of epochs

		Output:
		self.w		:	numpy array of shape (M,)	:	weights
		self.b		:	float						:	bias term
		'''

		y=np.array([])

		# print("Selection Started")
		'''
		the code in "if 0:" is the actual sample selection loop.
		the problem is that it takes ~140 seconds even without
		any logic written in the sample_selection() fn. hence
		i've written the "else" which takes all samples at once
		'''
		if 0:
			for curr_index, curr_sample in enumerate(train_data):
				if self.sample_selection(curr_sample):
					# y=1 if label is of class1, else -1
					y=np.append(y, 1 if train_label[curr_index]==self.c[0] else -1)
		else:
			for label in train_label:
				# y=1 if label is of class1, else -1
				y=np.append(y, 1 if label==self.c[0] else -1)
			self.train_set=train_data
		# print("Selection Done")

		x=np.append(self.train_set,np.ones([np.shape(self.train_set)[0],1]),axis=1)
		w=np.append(self.w,self.b)

		def grad_func(w,x,y):
			# calculate gradient
			diff=1-y*(np.dot(x,w))
			grad=np.zeros(len(w))
			for i,d in enumerate(diff):
				if max(0,d)!=0:
					dw=2*w*self.lamda-y[i]*x[i]
				else:
					dw=2*w*self.lamda
				grad=grad+dw
			grad=grad/np.shape(diff)[0]
			return grad

		for e in range(1, self.epochs):
			gradient=grad_func(w,x,y)
			w=w-self.alpha*gradient

		self.w=w[0:-1]
		self.b=w[-1]

	def f(self, input):
		# decision function based on g(y)
		dec=np.zeros(len(input))
		for index,value in enumerate(input):
			if value>=0:
				dec[index]=self.c[0]
			else:
				dec[index]=self.c[1]
		return dec

	def test(self, test_set):
		# return vector with classification decision
		g=np.dot(test_set,self.w)+self.b
		return self.f(g)