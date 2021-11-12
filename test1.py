from MyClassifier_5 import ifier
import numpy as np

filename='mnist_train.csv'
with open(filename, "r") as fid:
	data = np.loadtxt(fid, delimiter=",")
train_data = data[:,1:]
train_label = data[:,0]

filename='mnist_test.csv'
with open(filename, "r") as fid:
	data = np.loadtxt(fid, delimiter=",")
test_data = data[:,1:]
test_label = data[:,0]

choice1 = 1
choice2 = 7

select=(train_label==choice1)|(train_label==choice2)
train_data = train_data[select,:]
train_label = train_label[select]

select=(test_label==choice1)|(test_label==choice2)
test_data = test_data[select,:]
test_label = test_label[select]

t=ifier(choice1,choice2,np.shape(train_data)[1], alpha=0.1, lamda=0.1, epochs=70)
t.train(train_data,train_label)

test_pred=t.test(test_data)
error_count=np.count_nonzero((test_pred-test_label)!=0)
total_count=len(test_pred)
error_perc=error_count*100/total_count

print("Error percentage =",error_perc)