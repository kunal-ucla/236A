from MyClassifier_5 import ifier as classifier # using this for normalizing the data for now
from sklearn import preprocessing
import numpy as np


class tester:
	def __init__(self):
		self.loaded = 0

	def load_train(self, filename):
		with open(filename, "r") as fid:
			data = np.loadtxt(fid, delimiter=",")
		self.train_data_all = data[:,1:]
		self.train_label_all = data[:,0]

	def load_test(self, filename):
		with open(filename, "r") as fid:
			data = np.loadtxt(fid, delimiter=",")
		self.test_data_all = data[:,1:]
		self.test_label_all = data[:,0]

	def select_class(self, choice1, choice2):
		self.choice1 = choice1
		self.choice2 = choice2

		select = (self.train_label_all==choice1) | (self.train_label_all==choice2)
		self.train_data = self.train_data_all[select,:]
		self.train_label = self.train_label_all[select]

		select = (self.test_label_all==choice1) | (self.test_label_all==choice2)
		self.test_data = self.test_data_all[select,:]
		self.test_label = self.test_label_all[select]

	def train(self, normal=0, alpha=0.1, lamda=0.1, epochs=40):
		if normal:
			scaler = preprocessing.StandardScaler().fit(self.train_data)
			self.train_data = scaler.transform(self.train_data)
			self.test_data = scaler.transform(self.test_data)

		self.t = classifier(self.choice1,self.choice2,np.shape(self.train_data)[1], alpha=alpha, lamda=lamda, epochs=epochs)
		self.t.train(self.train_data,self.train_label)

	def test(self):
		train_pred = self.t.test(self.train_data)
		error_count = np.count_nonzero((train_pred-self.train_label)!=0)
		total_count = len(train_pred)
		error_perc = error_count*100/total_count
		print("Train Error percentage =",error_perc)

		test_pred = self.t.test(self.test_data)
		error_count = np.count_nonzero((test_pred-self.test_label)!=0)
		total_count = len(test_pred)
		error_perc = error_count*100/total_count
		print("Test Error percentage =",error_perc)

	def run(self, **kwargs):
		if "train_file" not in kwargs:
			kwargs["train_file"]='mnist_train.csv'
		if "test_file" not in kwargs:
			kwargs["test_file"]='mnist_test.csv'
		if "class1" not in kwargs:
			kwargs["class1"]=1
		if "class2" not in kwargs:
			kwargs["class2"]=7
		if "normal" not in kwargs:
			kwargs["normal"]=0
		if "alpha" not in kwargs:
			kwargs["alpha"]=0.1
		if "lambda" not in kwargs:
			kwargs["lambda"]=0.1
		if "epochs" not in kwargs:
			kwargs["epochs"]=50

		if self.loaded == 0:
			self.load_train(filename=kwargs["train_file"])
			self.load_test(filename=kwargs["test_file"])
			self.loaded = 1

		self.select_class(choice1=kwargs["class1"], choice2=kwargs["class2"])
		self.train(normal=kwargs["normal"], alpha=kwargs["alpha"], lamda=kwargs["lambda"], epochs=kwargs["epochs"])
		self.test()


