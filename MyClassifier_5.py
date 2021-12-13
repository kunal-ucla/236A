import numpy as np
import cvxpy as cp
import random

class ifier:
    def __init__(self, class1, class2, num_features, lamda=0.1, prob=0, init=10, skip=0, k=5, switch=0):
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
        self.k=k
        self.switch=switch

    def ILP(self, train_features, train_labels):

        # number of "positive" examples
        p = sum(c==self.c[0] for c in train_labels)
        # number of "negative" examples
        q = sum(c==self.c[1] for c in train_labels)
        # total number of samples
        n = p + q
        # KNN parameter
        k = self.k

        print('Computing D matrix', end='\r')
        # D matrix computation
        D = np.zeros((q,p))
        for i,v in enumerate(train_features[train_labels==self.c[1]]):
            for j,w in enumerate(train_features[train_labels==self.c[0]]):
                D[i,j] = np.linalg.norm(v-w)

        print('Computing Nn and Np', end='\r')
        # Nn computation
        Nn = np.zeros((q,k))
        for i in range(0,q):
            Nn[i,:] = np.argsort(D[i,:])[0:k]

        # Np computation
        Np = np.zeros((p,k))
        for j in range(0,p):
            Np[j,:] = np.argsort(D[:,j])[0:k]

        print('Computing Wp and Wn', end='\r')
        # Wp,Wn computation
        Wp = np.zeros((q,p))
        Wn = np.zeros((q,p))
        for i in range(0,q):
            for j in range(0,p):
                Wp[i,j] = 1 if j in Nn[i,:] else 0
                Wn[i,j] = 1 if i in Np[j,:] else 0

        print('Computing f_p and f_n', end='\r')
        # f_p, f_n computation
        f_p = np.zeros(p)
        f_n = np.zeros(q)
        for i in range(0,p):
            f_p[i] = sum(Wp[:,i])
        for i in range(0,q):
            f_n[i] = sum(Wn[i,:])

        print('Computations done ...')

        # ILP formulation
        f = np.concatenate([f_p,f_n])
        s = cp.Variable(p+q,integer=True)
        objective = cp.Minimize(cp.sum(s) - self.switch*f.T@s) # Min (sum(s[i]) - sum(f[i]*s[i]))
        constraints = []
        for i in range(0,n):
            constraints += [
                s[i] <= 1,
                s[i] >= 0
            ]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        ans = s.value
        selected = np.zeros(n)
        selected[train_labels==self.c[0]] = ans[0:p]
        selected[train_labels==self.c[1]] = ans[p:n]

        for i,v in enumerate(s.value):
            if selected[i] == 1:
                self.train_set=np.append(self.train_set, np.reshape(train_features[i],(1,len(train_features[i]))), axis=0)
                self.train_label=np.append(self.train_label, train_labels[i])

        return self.train_set,self.train_label

    def LP(self, train_features, train_labels):

        # number of "positive" examples
        p = sum(c==self.c[0] for c in train_labels)
        # number of "negative" examples
        q = sum(c==self.c[1] for c in train_labels)
        # total number of samples
        n = p + q
        # KNN parameter
        k = self.k

        print('Computing D matrix', end='\r')
        # D matrix computation
        D = np.zeros((q,p))
        for i,v in enumerate(train_features[train_labels==self.c[1]]):
            for j,w in enumerate(train_features[train_labels==self.c[0]]):
                D[i,j] = np.linalg.norm(v-w)

        print('Computing Nn and Np', end='\r')
        # Nn computation
        Nn = np.zeros((q,k))
        for i in range(0,q):
            Nn[i,:] = np.argsort(D[i,:])[0:k]

        # Np computation
        Np = np.zeros((p,k))
        for j in range(0,p):
            Np[j,:] = np.argsort(D[:,j])[0:k]

        print('Computing Wp and Wn', end='\r')
        # Wp,Wn computation
        Wp = np.zeros((q,p))
        Wn = np.zeros((q,p))
        for i in range(0,q):
            for j in range(0,p):
                Wp[i,j] = 1 if j in Nn[i,:] else 0
                Wn[i,j] = 1 if i in Np[j,:] else 0

        print('Computing f_p and f_n', end='\r')
        # f_p, f_n computation
        f_p = np.zeros(p)
        f_n = np.zeros(q)
        for i in range(0,p):
            f_p[i] = sum(Wp[:,i])
        for i in range(0,q):
            f_n[i] = sum(Wn[i,:])

        print('Computations done ...')

        # ILP formulation
        f = np.concatenate([f_p,f_n])
        s = cp.Variable(p+q)
        objective = cp.Minimize(cp.sum(s) - self.switch*f.T@s) # Min (sum(s[i]) - sum(f[i]*s[i]))
        constraints = []
        for i in range(0,n):
            constraints += [
                s[i] <= 1,
                s[i] >= 0
            ]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        ans = s.value
        selected = np.zeros(n)
        selected[train_labels==self.c[0]] = ans[0:p]
        selected[train_labels==self.c[1]] = ans[p:n]

        for i,v in enumerate(s.value):
            if selected[i] >= 0.5:
                selected[i] = 1
                self.train_set=np.append(self.train_set, np.reshape(train_features[i],(1,len(train_features[i]))), axis=0)
                self.train_label=np.append(self.train_label, train_labels[i])

        return self.train_set,self.train_label

    def sample_selection(self, training_sample, training_label):
        '''
        Input:
        training_sample     :    numpy array of shape (M,)  :   current sample
        training_label      :    int                        :   current label

        Output:
        self.train_set[]    :    numpy array of shape (N,M) :   append current_sample to this if selected
        self.train_label[]  :    numpy array of shape (N,)  :   append current_label to this if selected
        '''

        if self.prob != 0:
            # random sampling, prob(selecting) = "self.prob"
            is_selected = np.random.choice([0,1],p=[1-self.prob,self.prob])
        elif self.counter < self.init:
            # select the 1st "self.init" samples for a start
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
        train_data    :    numpy array of shape (N,M)   :   features
        train_label   :    numpy array of shape (N,)    :   labels
        self.lamda    :    float                        :   regularization param

        Output:
        self.w        :    numpy array of shape (M,)    :   weights
        self.b        :    float                        :   bias term
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
        objective = cp.Minimize(v1.T@t/np.shape(x)[0] + self.lamda*cp.norm(a-w,2)**2)
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
        np.savetxt("weights_5.csv", self.w, delimiter=",")
        np.savetxt("bias_5.csv", self.w, delimiter=",")

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