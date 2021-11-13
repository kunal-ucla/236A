import numpy as np

class ifier:
	def __init__(self, class1, class2, num_features, alpha=1, lamda=1, epochs=100):
		self.c=np.array([class1,class2])
		self.M=num_features
		self.train_set=np.array([])
		self.w=np.zeros(self.M)
		self.b=0
		self.alpha=alpha
		self.lamda=lamda
		self.epochs=epochs

	def sample_selection(self, training_sample):
		# select from training_sample[] and append to self.train_set[]
		return

	def train(self, train_data, train_label):
		'''
		Output:
		self.w		:	numpy array of shape (M,)	:	weights
		self.b		:	float						:	bias term

		Input:
		train_data	:	numpy array of shape (N,M)	:	features
		train_label	:	numpy array of shape (N,)	:	labels
		self.alpha	:	float						:	learning rate
		self.lamda	:	float						:	regularization param
		self.epochs	:	int							:	number of epochs
		'''

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

		y=np.zeros(len(train_label))
		y[train_label==self.c[0]]=1
		y[train_label==self.c[1]]=-1
		w=np.append(self.w,self.b)
		x=np.append(train_data,np.ones([np.shape(train_data)[0],1]),axis=1)

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