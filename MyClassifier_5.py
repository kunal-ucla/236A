import numpy as np
import cvxpy as cp
import random

class ifier:
    def __init__(self, class1, class2, num_features, lamda, prob, init, skip):
        self.c=np.array([class1,class2])
        self.M=num_features
        self.train_set=np.empty((0,num_features))
        self.train_label=np.array([])
        self.w=np.zeros(self.M)
        self.b=0
        self.lamda=lamda
        self.prob=prob
        self.init=init
        self.counter=0
        self.skip=skip

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
            # append the current training_sample and training_label to the train_set and train_label
            self.train_set=np.append(self.train_set, np.reshape(training_sample,(1,len(training_sample))), axis=0)
            self.train_label=np.append(self.train_label, training_label)
            # increment counter <== total number of training samples selected so far
            self.counter += 1
            if self.counter >= self.init:
                if self.counter == self.init:
                    # train the 1st "init" samples at once
                    self.train(self.train_set, self.train_label)
                elif self.counter % self.skip == 0:
                    # re-train only once every "skip" samples
                    self.train(self.train_set[-1:,:], self.train_label[-1:])

        return is_selected

    def train(self, train_data, train_label):
        '''
        Input:
        train_data    :    numpy array of shape (N,M)   :    features
        train_label   :    numpy array of shape (N,)    :    labels
        self.lamda    :    float                        :    regularization param

        Output:
        self.w        :    numpy array of shape (M,)    :    weights
        self.b        :    float                        :    bias term
        '''

        y = np.zeros(len(train_label))
        y[train_label==self.c[0]] = 1
        y[train_label==self.c[1]] = -1
        w = np.append(self.w,self.b)
        x = np.append(train_data,np.ones([np.shape(train_data)[0],1]),axis=1)

        #  --------- LP start ---------  #
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
        #  --------- LP end  ---------  #

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