import numpy as np

class ifier:
	def __init__(self, class1, class2, num_features):
		self.c=np.array([class1,class2])
		self.M=num_features
		self.train_set=np.array([])
		self.w=np.zeros(self.M)
		self.b=0

	def sample_selection(self, training_sample):
		# select from training_sample[] and append to self.train_set[]
		return

	def train(self, train_data, train_label):
		# fill self.w and self.b
		lamda=1
		alpha=1

		def cost_func(w,x,y):
			# calculate total error for a given w
			diff=1-y*(np.dot(x,w))
			diff[diff<0]=0
			penalty=lamda*np.dot(w,w)+np.sum(diff)/np.shape(diff)[0]
			return penalty

		def grad_func(w,x,y):
			# calculate gradient
			diff=1-y*(np.dot(x,w))
			grad=np.zeros(len(w))
			for i,d in enumerate(diff):
				if max(0,d)!=0:
					grad=grad-y[i]*x[i]
			grad=2*lamda*w+grad/np.shape(diff)[0]
			return grad

		y=np.zeros(len(train_label))
		y[train_label==self.c[0]]=1
		y[train_label==self.c[1]]=-1
		w=np.append(self.w,self.b)
		x=np.append(train_data,np.ones([np.shape(train_data)[0],1]),axis=1)

		epochs=100
		for e in range(1, epochs):
			gradient=grad_func(w,x,y)
			w=w-alpha*gradient

		self.w=w[0:-1]
		self.b=w[-1]
		return

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