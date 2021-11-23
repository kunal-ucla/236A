from MyClassifier_5 import ifier as classifier # using this for normalizing the data for now
from sklearn import preprocessing
import numpy as np
from numpy.random import normal
import time
import sys
import matplotlib.pyplot as plt
import matplotlib

class tester:
    def __init__(self, syn=0):
        self.loaded = 0
        self.syn = syn
        self.size = 500

    def load_train(self, filename):
        if self.syn == 1:
            x=np.transpose(normal(loc=[-1,1],scale=1,size=[4*self.size ,2]))    # class 1
            y=np.transpose(normal(loc=[1,-1],scale=1,size=[4*self.size ,2]))    # class -1
            choice = np.random.randint(2, size = np.shape(x)[1]).astype(bool)
            self.train_data_all = np.transpose(np.where(choice, x, y))
            self.train_label_all = np.where(choice, 1, -1)
        else:
            with open(filename, "r") as fid:
                data = np.loadtxt(fid, delimiter=",")
            self.train_data_all = data[:,1:]
            self.train_label_all = data[:,0]
        self.loaded = min(2,1+self.loaded)

    def load_test(self, filename):
        if self.syn == 1:
            x=np.transpose(normal(loc=[-1,1],scale=1,size=[self.size ,2]))    # class 1
            y=np.transpose(normal(loc=[1,-1],scale=1,size=[self.size ,2]))    # class -1
            choice = np.random.randint(2, size = np.shape(x)[1]).astype(bool)
            self.test_data_all = np.transpose(np.where(choice, x, y))
            self.test_label_all = np.where(choice, 1, -1)
        else:
            with open(filename, "r") as fid:
                data = np.loadtxt(fid, delimiter=",")
            self.test_data_all = data[:,1:]
            self.test_label_all = data[:,0]
        self.loaded = min(2,1+self.loaded)

    def select_class(self, choice1, choice2):
        if self.syn == 1:
            self.choice1 = 1
            self.choice2 = -1
            self.train_data = self.train_data_all
            self.train_label = self.train_label_all
            self.test_data = self.test_data_all
            self.test_label = self.test_label_all
        else:
            self.choice1 = choice1
            self.choice2 = choice2

            select = (self.train_label_all==choice1) | (self.train_label_all==choice2)
            self.train_data = self.train_data_all[select,:]
            self.train_label = self.train_label_all[select]

            select = (self.test_label_all==choice1) | (self.test_label_all==choice2)
            self.test_data = self.test_data_all[select,:]
            self.test_label = self.test_label_all[select]

    def train(self, normal, alpha, lamda, epochs, method, select, prob, init, skip, stop, last):
        if normal:
            scaler = preprocessing.StandardScaler().fit(self.train_data)
            self.train_data = scaler.transform(self.train_data)
            self.test_data = scaler.transform(self.test_data)

        self.num_features = np.shape(self.train_data)[1]
        self.t = classifier(self.choice1,self.choice2, self.num_features, alpha=alpha, lamda=lamda, epochs=epochs, method=method, prob=prob, init=init, skip=skip, last=last)
        
        start = time.time()
        if select:
            print("Selecting samples...")
            chosen = 0
            total = 0
            i = 0
            accu = np.array([50, 65, 80, 95])
            for curr_sample,curr_label in zip(self.train_data,self.train_label):
                curr_chosen = self.t.sample_selection(curr_sample,curr_label)
                chosen = chosen + curr_chosen
                total = total + 1
                print('Selected %d out of %d samples' %(chosen,total), end='\r')
                # if (curr_chosen == 1) & (i < len(accu)):
                #     if (self.error(self.train_data,self.train_label) < 100 - accu[i]):
                #         print('Reached %f %% accuracy at %d samples' %(accu[i], chosen))
                #         i = i + 1
                if (curr_chosen == 1) & (stop != 0):
                    if self.error(self.train_data[total:,:],self.train_label[total:]) < stop:
                        break
            self.selected_data = self.t.train_set
            self.selected_label = self.t.train_label
        else:
            chosen = np.shape(self.train_data)[0]
            self.selected_data = self.train_data
            self.selected_label = self.train_label
        end = time.time()
        print("Time taken for selection = %f\n" %(end-start))
        print('Sending %d samples for training...' %(chosen))

        # reset weights in case sample_selection set some weights, and train afresh on selected samples
        # self.t.w = np.zeros(self.t.M)
        # self.t.b = 0
        if method == 'gd':
        # train with 50 epochs at the end in case of GD
            self.t.epochs = 50
        
        start = time.time()
        self.t.train(self.selected_data,self.selected_label)
        end = time.time()
        print("Time taken for training = %f\n" %(end-start))

    def error(self, data, label):
        pred = self.t.test(data)
        error_count = np.count_nonzero((pred-label)!=0)
        total_count = len(pred)
        error_perc = error_count*100/total_count
        return error_perc

    def test(self):
        print("Starting tests...")

        print("Train Error percentage =",self.error(self.train_data,self.train_label))

        print("Test Error percentage =",self.error(self.test_data,self.test_label))

    def plot(self):
        colors=['red','blue']
        y=self.train_label
        y[y==-1]=0
        plt.figure(1)
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.scatter(self.train_data[:,0],self.train_data[:,1],c=y,cmap=matplotlib.colors.ListedColormap(colors))

        colors=['red','blue']
        y=self.selected_label
        y[y==-1]=0
        plt.figure(2)
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.scatter(self.selected_data[:,0],self.selected_data[:,1],c=y,cmap=matplotlib.colors.ListedColormap(colors))
        
        plt.show()

    def run(self, **kwargs):
        if "train_file" not in kwargs:
            kwargs["train_file"]='mnist_train.csv'
        else:
            self.loaded=0
        if "test_file" not in kwargs:
            kwargs["test_file"]='mnist_test.csv'
        else:
            self.loaded=0
        if "class1" not in kwargs:
            kwargs["class1"]=1
        if "class2" not in kwargs:
            kwargs["class2"]=7
        if "normal" not in kwargs:
            kwargs["normal"]=0
        if "alpha" not in kwargs:
            kwargs["alpha"]=0.1
        if "lamda" not in kwargs:
            kwargs["lamda"]=0.1
        if "epochs" not in kwargs:
            kwargs["epochs"]=10
        if "method" not in kwargs:
            kwargs["method"]='gd'
        elif kwargs["method"]=='lp':
            kwargs["epochs"]=1
            if "last" not in kwargs:
                kwargs["last"]=1
        if "last" not in kwargs:
            kwargs["last"]=20
        if "select" not in kwargs:
            kwargs["select"]=0
        if "prob" not in kwargs:
            kwargs["prob"]=0
        if "init" not in kwargs:
            kwargs["init"]=2
        if "skip" not in kwargs:
            kwargs["skip"]=1
        if "stop" not in kwargs:
            kwargs["stop"]=0


        if self.loaded != 2:
            self.load_train(filename=kwargs["train_file"])
            self.load_test(filename=kwargs["test_file"])
            self.loaded = 2

        self.select_class(choice1=kwargs["class1"], choice2=kwargs["class2"])
        self.train(normal=kwargs["normal"], alpha=kwargs["alpha"], lamda=kwargs["lamda"], epochs=kwargs["epochs"], method=kwargs["method"], select=kwargs["select"], prob=kwargs["prob"], init=kwargs["init"], skip=kwargs["skip"], stop=kwargs["stop"], last=kwargs["last"])
        self.test()

if len(sys.argv)>1:
    if sys.argv[1]=='-s':
        test = tester(1)
        if len(sys.argv)>2:
            test.size = int(sys.argv[2])
        print("Loading synthetic datasets...")
    else:
        test = tester()
        print("Loading default datasets...")
else:
    test = tester()
    print("Loading default datasets...")
test.load_train('mnist_train.csv')
test.load_test('mnist_test.csv')
print("Ready. Start by running 'test.run()'")
