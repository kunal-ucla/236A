from MyClassifier_5 import ifier as classifier # using this for normalizing the data for now
from sklearn import preprocessing
import numpy as np
from numpy.random import normal
import time
import sys
import getopt
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans

class tester:
    def __init__(self, syn_size, live_plot, train_file, test_file):
        
        self.syn_size = syn_size
        self.live_plot = live_plot
        self.train_file = train_file
        self.test_file = test_file

        print("Loading train dataset...", end='\r')
        if self.syn_size != 0:
            x=np.transpose(normal(loc=[-1,1],scale=1,size=[4*self.syn_size ,2]))    # class 1
            y=np.transpose(normal(loc=[1,-1],scale=1,size=[4*self.syn_size ,2]))    # class -1
            choice = np.random.randint(2, size = np.shape(x)[1]).astype(bool)
            self.train_data_all = np.transpose(np.where(choice, x, y))
            self.train_label_all = np.where(choice, 1, -1)
        else:
            with open(train_file, "r") as fid:
                data = np.loadtxt(fid, delimiter=",")
            self.train_data_all = data[:,1:]
            self.train_label_all = data[:,0]

        print("Loading test dataset...", end='\r')
        if self.syn_size != 0:
            x=np.transpose(normal(loc=[-1,1],scale=1,size=[self.syn_size ,2]))    # class 1
            y=np.transpose(normal(loc=[1,-1],scale=1,size=[self.syn_size ,2]))    # class -1
            choice = np.random.randint(2, size = np.shape(x)[1]).astype(bool)
            self.test_data_all = np.transpose(np.where(choice, x, y))
            self.test_label_all = np.where(choice, 1, -1)
        else:
            with open(test_file, "r") as fid:
                data = np.loadtxt(fid, delimiter=",")
            self.test_data_all = data[:,1:]
            self.test_label_all = data[:,0]

        print(" "*24, end='\r')
        print("Datasets loaded.")

    def select_class(self, choice1, choice2):
        if self.syn_size != 0:
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
        self.num_features = np.shape(self.train_data)[1]

    def train(self, normal, lamda, select, prob, init, skip, stop, step):
        if normal:
            scaler = preprocessing.StandardScaler().fit(self.train_data)
            self.train_data = scaler.transform(self.train_data)
            self.test_data = scaler.transform(self.test_data)
        
        self.t = classifier(self.choice1, self.choice2, self.num_features, lamda=lamda, prob=prob, init=init, skip=skip)
        
        start = time.time()
        if select == 1:
            print("Selecting samples...")
            chosen = 0
            total = 0
            i = 0
            accu = np.array([50, 65, 80, 90, 95])

            # live plot start
            if self.live_plot:
                colors=['red','blue']
                plt.ion()
                plt.xlim(-4, 4)
                plt.ylim(-4, 4)
                plt.title("Selected Training Samples")
                x1 = np.linspace(-4,4,100)
                x2 = x1*0
                graph = plt.plot(x1,x2,'-g')[0]
            # live plot end

            for curr_sample,curr_label in zip(self.train_data,self.train_label):
                curr_chosen = self.t.sample_selection(curr_sample,curr_label)
                chosen = chosen + curr_chosen
                total = total + 1
                print('Selected %d out of %d samples' %(chosen,total), end='\r')
                if step:
                    if (curr_chosen == 1) & (i < len(accu)):
                        if (self.error(self.train_data,self.train_label) < 100 - accu[i]):
                            print('Reached %f %% accuracy at %d samples' %(accu[i], chosen))
                            i = i + 1
                if (curr_chosen == 1) & (stop != 0):
                    if self.error(self.train_data[total:,:],self.train_label[total:]) < stop:
                        break
                # live plot start
                if curr_chosen & self.live_plot:
                    curr_label = 0 if curr_label==-1 else 1
                    plt.scatter(curr_sample[0],curr_sample[1],color=colors[curr_label])
                    if self.t.w[0]!= 0:
                        x2 = - (self.t.w[1]/self.t.w[0])*x1 - (self.t.b/self.t.w[0])
                    graph.set_ydata(x2)
                    plt.draw()
                    plt.pause(1e-10)
                # live plot end
            self.selected_data = self.t.train_set
            self.selected_label = self.t.train_label
            self.chosen = chosen
        elif select == 2:
            print("Selecting samples...")
            # total_chosen = self.t.cluster_selection(self.train_data,self.train_label,self.chosen)
            total_chosen = self.t.ILP_selection_2(self.train_data,self.train_label)
            chosen = sum(total_chosen)
            self.selected_data = self.t.train_set
            self.selected_label = self.t.train_label
            self.temp = total_chosen
        else:
            chosen = np.shape(self.train_data)[0]
            self.selected_data = self.train_data
            self.selected_label = self.train_label
        end = time.time()
        print("Time taken for selection = %f\n" %(end-start))
        print('Sending %d samples for training...' %(chosen))
        
        start = time.time()
        self.t.train(self.selected_data,self.selected_label)
        end = time.time()
        print("Time taken for training = %f\n" %(end-start))
        # live plot start
        if (select==1) & (self.live_plot):
            plt.ioff()
            input("Press Enter to close graph...")
            plt.close()
        # live plot end

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
        x1 = np.linspace(-4,4,100)
        x2 = - (self.t.w[1]/self.t.w[0])*x1 - (self.t.b/self.t.w[0])

        self.plot_scatter(self.selected_data,self.selected_label,title="Selected Training Samples")
        plt.plot(x1,x2,'-g')

    def plot_scatter(self, features, labels, show = False, title = None, i = [0]):
        i[0] = i[0] + 1
        plt.figure(i[0])
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.scatter(features[:,0], features[:,1], c = labels) 
        if title is not None:
            plt.title(title)
        if show:
            plt.show() 

    def run(self, **kwargs):

        if "class1" not in kwargs:
            kwargs["class1"]=1
        if "class2" not in kwargs:
            kwargs["class2"]=7
        if "normal" not in kwargs:
            kwargs["normal"]=0
        if "lamda" not in kwargs:
            kwargs["lamda"]=0.1
        if "select" not in kwargs:
            kwargs["select"]=2
        if "prob" not in kwargs:
            kwargs["prob"]=0
        if "init" not in kwargs:
            kwargs["init"]=2
        if "skip" not in kwargs:
            kwargs["skip"]=1
        if "stop" not in kwargs:
            kwargs["stop"]=0
        if "step" not in kwargs:
            kwargs["step"]=0

        self.select_class(choice1=kwargs["class1"], choice2=kwargs["class2"])

        if self.syn_size != 0:
            # self.plot_scatter(self.train_data,self.train_label,title="Full Training Samples")
            # self.plot_scatter(self.test_data,self.test_label,title="Testing Samples")

            print('='*20)
            print("Running Part 1")
            print('='*20)
            self.train(normal=kwargs["normal"], lamda=kwargs["lamda"], select=1, prob=kwargs["prob"], init=kwargs["init"], skip=kwargs["skip"], stop=kwargs["stop"], step=kwargs["step"])
            self.test()

        if self.syn_size != 0:
            self.plot()

        print('='*20)
        print("Running Part 2")
        print('='*20)
        self.train(normal=kwargs["normal"], lamda=kwargs["lamda"], select=2, prob=kwargs["prob"], init=kwargs["init"], skip=kwargs["skip"], stop=kwargs["stop"], step=kwargs["step"])
        self.test()

        if self.syn_size != 0:
            self.plot()
            plt.show()

try:
    opts, args = getopt.getopt(sys.argv[1:], "s:p", ["train=","test="])
except getopt.GetoptError as err:
    print(err)
    print("Usage: 'python -i test1.py'")
    print("Optional Flags: -s <test_size_int>, -p, --test_file <test_file_name>, --train_file <train_file_name>")
    sys.exit(2)

syn_size = 0
live_plot = 0
test_file = "mnist_train.csv"
train_file = "mnist_test.csv"

for o, a in opts:
    if o == "-s":
        syn_size = int(a)
    elif o == "-p":
        live_plot = 1
    elif o == "--test":
        test_file = a
    elif o == "--train":
        train_file = a

test = tester(syn_size,live_plot,train_file,test_file)
print("Ready. Start by running 'test.run()'")

