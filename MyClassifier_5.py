import numpy as np
import random

class ifier:
	def __init__(self, class1, class2, num_features, alpha, lamda, epochs, method, shuffle, prob, init, delta, limit):
		self.c=np.array([class1,class2])
		self.M=num_features
		self.train_set=np.empty((0,num_features))
		self.train_label=np.array([])
		self.w=np.zeros(self.M)
		self.b=0
		self.alpha=alpha
		self.lamda=lamda
		self.epochs=epochs
		self.method=method
		self.shuffle=shuffle
		self.prob=prob
		self.init=init
		self.delta=delta

		# test
		self.counter=0
		self.limit=limit

	def sample_selection(self, training_sample, training_label):
		'''
		Input:
		training_sample		:	numpy array of shape (M,)	:	current sample
		training_label		:	int							:	current label

		Output:
		self.train_set[]	:	numpy array of shape (N,M)	:	append current_sample to this if selected
		self.train_label[]	:	numpy array of shape (N,)	:	append current_label to this if selected
		'''

		# code here for checking whether curr_sample should be included
		if self.prob==0:
			if np.shape(self.train_set)[0] < self.init:
				is_selected=1 # select the 1st 100(self.init) samples for sure
				self.counter=self.limit
			else:
				if self.counter==self.limit:
					self.train(self.train_set, self.train_label)
					self.counter=0
				else:
					self.counter+=1
				x=np.append(training_sample,1)
				w=np.append(self.w,self.b)
				y=1 if training_label==self.c[0] else -1
				p=np.dot(x,w)

				if abs(p) < self.delta:
					is_selected=1
				else:
					is_selected=0
		else:
			is_selected=np.random.choice([0,1],p=[1-self.prob,self.prob])

		if is_selected:
			self.train_set=np.append(self.train_set, np.reshape(training_sample,(1,len(training_sample))), axis=0)
			self.train_label=np.append(self.train_label, training_label)
		
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

		def grad_func(w,x,y):
			'''
			Input:
			w,x,y

			Output:
			gradient
			'''
			
			# if only 1 example is passed (SGD)
			if type(y) == np.float64:
				y = np.array([y])
				x = np.array([x])

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

		y=np.zeros(len(train_label))
		y[train_label==self.c[0]]=1
		y[train_label==self.c[1]]=-1
		w=np.append(self.w,self.b)
		x=np.append(train_data,np.ones([np.shape(train_data)[0],1]),axis=1)

		for e in range(0, self.epochs):
			if self.method=='sgd':
				shuffle=list(range(0,np.shape(y)[0]))
				if self.shuffle:
					random.shuffle(shuffle) # shuffle at every epoch
				for i in shuffle:
					gradient=grad_func(w,x[i,:],y[i])
					w=w-self.alpha*gradient
			elif self.method=='gd':
				gradient=grad_func(w,x,y)
				w=w-self.alpha*gradient
			elif self.method=='bgd':
				end=np.shape(y)[0]
				start=max(0,end-20) # use last 20 samples at every step
				gradient=grad_func(w,x[start:end,:],y[start:end])
				w=w-self.alpha*gradient
			else:
				print("Invalid method, try again")

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