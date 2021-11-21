import numpy as np
import cvxpy as cp
import random

class ifier:
    def __init__(self, class1, class2, num_features, alpha, lamda, epochs, method, prob, init, skip, min_cos, max_cos, avg_cos, last):
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
        self.prob=prob
        self.init=init
        self.counter=0
        self.skip=skip
        self.last=last

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

        if self.prob != 0:
        # random sampling with probability of selecting "self.prob"
            is_selected = np.random.choice([0,1],p=[1-self.prob,self.prob])
        elif self.counter < self.init:
        # select the 1st "self.init" samples for sure
            is_selected = 1
        elif abs(np.dot(training_sample,self.w)+self.b) < 1:
        # points within 1 margin from current hyperplane
            is_selected = 1
        else:
        # otherwise don't select the point
            is_selected = 0

        if is_selected:
            self.train_set=np.append(self.train_set, np.reshape(training_sample,(1,len(training_sample))), axis=0)
            self.train_label=np.append(self.train_label, training_label)
            self.counter += 1
        if self.counter >= self.init:
            if self.counter == self.init:
            # train the 1st "init" samples at once
                self.train(self.train_set, self.train_label)
            elif (self.counter % self.skip == 0) & ((self.method != 'lp') | is_selected):
            # skip re-training every "skip" samples; re-train only if a sample is selected in LP case
                self.train(self.train_set[-self.last:,:], self.train_label[-self.last:])

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

        y = np.zeros(len(train_label))
        y[train_label==self.c[0]] = 1
        y[train_label==self.c[1]] = -1
        w = np.append(self.w,self.b)
        x = np.append(train_data,np.ones([np.shape(train_data)[0],1]),axis=1)

        for e in range(0, self.epochs):
            if self.method == 'gd':
                gradient = self.grad_func(w,x,y)
                w = w - self.alpha*gradient
            elif self.method=='lp':
                t = cp.Variable(np.shape(x)[0])
                a = cp.Variable(np.shape(x)[1])
                v1 = np.ones(np.shape(x)[0])
                objective = cp.Minimize(v1.T@t/np.shape(x)[0] + self.lamda*cp.norm(a-w,1))
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

        self.w = w[0:-1]
        self.b = w[-1]

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