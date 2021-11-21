import numpy as np
import cvxpy as cp
import random

class ifier:
    def __init__(self, class1, class2, num_features, alpha, lamda, epochs, method, shuffle, prob, init, delta, skip, min_cos, max_cos, avg_cos):
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
        self.counter=0
        self.skip=skip
        self.min_cos=min_cos
        self.max_cos=max_cos
        self.avg_cos=avg_cos
        self.corr_sum=np.zeros(self.M)
        self.selected=0

    def cos(self, sample1, sample2):
        # calculates cos(angle) between 2 points in the dimension space
        num = abs(np.dot(sample1,sample2))
        den = np.sqrt(np.dot(sample1,sample1)*np.dot(sample2,sample2))
        cos = num/den
        return cos

    def grad_func(self, w, x, y):
        # calculates gradient
        if type(y) == np.float64:   # if only 1 example is passed (SGD)
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

    def sample_selection(self, training_sample, training_label):
        '''
        Input:
        training_sample     :    numpy array of shape (M,)    :    current sample
        training_label      :    int                          :    current label

        Output:
        self.train_set[]    :    numpy array of shape (N,M)   :    append current_sample to this if selected
        self.train_label[]  :    numpy array of shape (N,)    :    append current_label to this if selected
        '''

        # code here for checking whether curr_sample should be included
        if self.prob==0:
            if np.shape(self.train_set)[0] < self.init:
                is_selected=1 # select the 1st 10(self.init) samples for sure
                self.counter=self.skip
            else:
                if self.counter==self.skip:
                    if self.method=='lp':
                        if self.selected:
                            self.train(self.train_set, self.train_label)
                        self.selected=0
                    else:
                        temp1=self.method
                        temp2=self.epochs
                        self.method='bgd' # always use bgd for re-train in sample selection
                        self.epochs=10
                        self.train(self.train_set, self.train_label)
                        self.method=temp1
                        self.epochs=temp2
                    self.counter=0
                else:
                    self.counter+=1
                x=np.append(training_sample,1)
                w=np.append(self.w,self.b)
                y=1 if training_label==self.c[0] else -1
                p=np.dot(x,w)

                if (abs(p) < self.delta): # | ((abs(p) > self.delta)&(p*y < 0)):
                    is_selected=1
                else:
                    is_selected=0
                if ((self.min_cos!=1)|((1==0) & (self.avg_cos!=1))|(self.max_cos!=1)) & (is_selected==1):
                    min_cos = 1
                    max_cos = 0
                    avg_cos = 0
                    for prev_sample in self.train_set:
                        cur_cos = self.cos(training_sample, prev_sample)
                        min_cos = min(min_cos, cur_cos)
                        max_cos = max(max_cos, cur_cos)
                        avg_cos = avg_cos + cur_cos
                    avg_cos = avg_cos/np.shape(self.train_set)[0]
                    if (min_cos > self.min_cos) | (max_cos > self.max_cos) | (avg_cos > self.avg_cos):
                        is_selected = 0
                if (self.avg_cos!=1) & (is_selected==1):
                    corr_curr = np.dot(self.corr_sum,training_sample)/np.sqrt(np.dot(training_sample,training_sample))
                    if abs(corr_curr) > self.avg_cos:
                        is_selected = 0
        else:
            is_selected=np.random.choice([0,1],p=[1-self.prob,self.prob])

        if is_selected:
            self.train_set=np.append(self.train_set, np.reshape(training_sample,(1,len(training_sample))), axis=0)
            self.train_label=np.append(self.train_label, training_label)
            num_selected = np.shape(self.train_set)[0]
            corr_new = training_sample/np.sqrt(np.dot(training_sample,training_sample))
            self.corr_sum = self.corr_sum*(num_selected-1) + corr_new
            self.corr_sum /= num_selected
            self.selected = 1

        return is_selected

    def train(self, train_data, train_label):
        '''
        Input:
        train_data    :    numpy array of shape (N,M)   :    features
        train_label   :    numpy array of shape (N,)    :    labels
        self.alpha    :    float                        :    learning rate
        self.lamda    :    float                        :    regularization param
        self.epochs   :    int                          :    number of epochs

        Output:
        self.w        :    numpy array of shape (M,)    :    weights
        self.b        :    float                        :    bias term
        '''

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
                    gradient=self.grad_func(w,x[i,:],y[i])
                    w=w-self.alpha*gradient
            elif self.method=='gd':
                gradient=self.grad_func(w,x,y)
                w=w-self.alpha*gradient
            elif self.method=='bgd':
                end=np.shape(y)[0]
                start=max(0,end-20) # use last 20 samples at every step
                gradient=self.grad_func(w,x[start:end,:],y[start:end])
                w=w-self.alpha*gradient
            elif self.method=='lp':
                t = cp.Variable(np.shape(x)[0])
                a = cp.Variable(np.shape(x)[1])
                v1 = np.ones(np.shape(x)[0])
                objective = cp.Minimize(v1.T@t)
                constraints = []
                for i in range(0,np.shape(x)[0]):
                    constraints += [
                        t[i] >= 0,
                        1-y[i]*(x[i].T@a) <= t[i]
                    ]
                prob = cp.Problem(objective, constraints)
                prob.solve()
                w = a.value
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