import numpy as np
import cvxpy as cp
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import statistics

class ifier:
    def __init__(self, class1, class2, num_features, lamda=0.1, prob=0, init=10, skip=0, csize=5):
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
        self.csize=csize

    def plot_scatter(self, features, labels, show = False, title = None, i = [0]):
        i[0] = i[0] + 1
        plt.figure(20+i[0])
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.scatter(features[:,0], features[:,1], c = labels) 
        if title is not None:
            plt.title(title)
        if show:
            plt.show() 

    def ILP_selection_too_slow(self, train_data, train_labels):
        '''
        Input:
        train_data      :   numpy array of shape (N,M)  :   entire training set
        train_labels        :   numpy array of shape (N,)   :   entire training labels

        Output:
        self.train_set[]    :   numpy array of shape (n,M)  :   selected training samples
        self.train_label[]  :   numpy array of shape (n,)   :   selected training labels
        '''
        
        x = np.append(train_data,np.ones([np.shape(train_data)[0],1]),axis=1)
        x0 = x[train_labels==self.c[0],:]
        x1 = x[train_labels==self.c[1],:]
        n, m = np.shape(x)
        n0 = np.shape(x0)[0]
        n1 = np.shape(x1)[0]
        y = np.zeros(n)
        y[train_labels==self.c[0]] = 1
        y[train_labels==self.c[1]] = -1

        print("Computing Mapper..")

        K = np.zeros((n,n))
        # for i in range(0,n):
        #     for j in range(i,n):
        #         K[i,j] = np.dot(x[i,:],x[j,:])
        #         K[j,i] = K[i,j]
        # A = np.zeros((n,n))
        # for i in range(0,n):
        #     for j in range(i,n):
        #         A[i,j] = K[i,j]/np.sqrt(K[i,i]*K[j,j])
        #         A[j,i] = A[i,j]
        D = np.zeros((n,n))
        for i in range(0,n):
            for j in range(i+1,n):
                D[i,j] = np.linalg.norm(x[i,:]-x[j,:])
                D[j,i] = D[i,j]

        C = np.zeros((n0,n1))
        for i in range(0,n0):
            for j in range(0,n1):
                C[i,j] = 1 # stopped here TODO

        #  --------- ILP start - part 2 ---------  #
        print("Constructing LP...")

        # ###### ToDo Start ######
        s = cp.Variable(n, integer=True) # selection bool for each sample
        objective = cp.Maximize(cp.sum(s))
        constraints = []
        for i in range(0,n):
            constraints += [
                s[i] <= 1,
                s[i] >= 0
            ]
        for i in range(0,n-1):
            for j in range(i+1,n):
                if y[i]*y[j] < 0:
                    constraints += [
                        D[i,j]*(s[i]+s[j]-1) <= 0.5
                    ]
                else:
                    constraints += [
                        D[i,j] >= 0.5*(s[i]+s[j]-1)
                    ]
        # ###### ToDo End ######

        print("Solving started...")

        prob = cp.Problem(objective, constraints)
        prob.solve()
        #  --------- ILP end - part 2 ---------  #


        # #  --------- ILP start ---------  #
        # t = cp.Variable(np.shape(x)[0]) # slack variable for max(0,1-y(w.x+b))
        # a = cp.Variable(np.shape(x)[1]) # weight variable w
        # s = cp.Bool(np.shape(x)[0]) # selection bool for each sample
        # v1 = np.ones(np.shape(x)[0])

        # ###### ToDo Start ######
        # objective = cp.Minimize()
        # constraints = []
        # ###### ToDo End ######

        # prob = cp.Problem(objective, constraints)
        # prob.solve(solver=cvxpy.GLPK_MI)
        # #  --------- ILP end  ---------  #

        # self.train_set = train_data[s.value,:]
        # self.train_label = train_labels[s.value]

        return s.value

    def cluster_selection_failed(self, train_features, train_labels):
        '''
        Input:
        train_features      :   numpy array of shape (N,M)  :   entire training set
        train_labels        :   numpy array of shape (N,)   :   entire training labels

        Output:
        self.train_set[]    :   numpy array of shape (n,M)  :   selected training samples
        self.train_label[]  :   numpy array of shape (n,)   :   selected training labels
        '''
        NUM_C = 10
        clf = KMeans(n_clusters=NUM_C, init="random", n_init=1, random_state=2)
        clf.fit(train_features[train_labels==self.c[0],:])
        clusters = clf.predict(train_features[train_labels==self.c[0],:])
        c1 = clf.cluster_centers_

        clf.fit(train_features[train_labels==self.c[1],:])
        clusters = clf.predict(train_features[train_labels==self.c[1],:])
        c2 = clf.cluster_centers_

        sorter = np.zeros((NUM_C*NUM_C,3))
        for i1,x1 in enumerate(c1):
            for i2,x2 in enumerate(c2):
                sorter[NUM_C*i1+i2,0] = np.linalg.norm(x1-x2)
                sorter[NUM_C*i1+i2,1] = i1
                sorter[NUM_C*i1+i2,2] = i2
        sorter = sorted(sorter, key = lambda x: x[0])

        x = np.concatenate([c1,c2],axis=0)
        y = [0]*NUM_C + [1]*NUM_C

        select_num = int(NUM_C/2) + 1
        for i in range(0,select_num):
            y[int(sorter[i][1])] = 2
            y[int(sorter[i][2])+NUM_C] = 2

        centers = np.zeros((3,np.shape(x)[1]))
        count = np.zeros(3)
        for i,v in enumerate(x):
            centers[y[i],:] += v
            count[y[i]] += 1
        for i in range(0,3):
            centers[i,:] /= count[i]

        selected = np.zeros(np.shape(train_labels))
        for i,v in enumerate(train_features):
            if np.linalg.norm(centers-v,axis=1).argmin() == 2:
                selected[i] = 1
                self.train_set=np.append(self.train_set, np.reshape(v,(1,len(v))), axis=0)
                self.train_label=np.append(self.train_label, train_labels[i])

        return selected

    def ILP_latest(self, train_features, train_labels):
        '''
        Input:
        train_features      :   numpy array of shape (N,M)  :   entire training set
        train_labels        :   numpy array of shape (N,)   :   entire training labels

        Output:
        self.train_set[]    :   numpy array of shape (n,M)  :   selected training samples
        self.train_label[]  :   numpy array of shape (n,)   :   selected training labels
        '''
        n = np.shape(train_labels)[0]
        NUM_C = int(n/self.csize)
        clf = KMeans(n_clusters=NUM_C, init="random", n_init=1, random_state=2)

        clf.fit(train_features[train_labels==self.c[0],:])
        clusters_1 = clf.predict(train_features[train_labels==self.c[0],:])
        centers_1 = clf.cluster_centers_

        clf.fit(train_features[train_labels==self.c[1],:])
        clusters_2 = clf.predict(train_features[train_labels==self.c[1],:])
        centers_2 = clf.cluster_centers_

        # cluster id
        cid = np.zeros(n)
        cid[train_labels==self.c[0]] = clusters_1
        cid[train_labels==self.c[1]] = clusters_2 + NUM_C

        # distance opposite class
        doc = np.zeros(n)
        for i,v in enumerate(train_features):
            opp_centers = centers_2 if train_labels[i] == self.c[0] else centers_1
            doc[i] = np.linalg.norm(opp_centers-v,axis=1).min()

        # distance same class
        dsc = np.zeros(n)
        for i,v in enumerate(train_features):
            neighbors = train_features[cid==cid[i]]
            dsc[i] = np.linalg.norm(neighbors-v,axis=1).min()

        # ILP formulation
        s = cp.Variable(n)
        objective = cp.Minimize(doc.T@s - dsc.T@s)
        constraints = [ cp.sum(s) >= int(n * 0.025 + 100*(1 - np.exp(-n))) ]
        for i in range(0,n):
            constraints += [
                s[i] <= 1,
                s[i] >= 0
            ]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        for i,v in enumerate(s.value):
            if v >= 0.5:
                self.train_set=np.append(self.train_set, np.reshape(train_features[i],(1,len(train_features[i]))), axis=0)
                self.train_label=np.append(self.train_label, train_labels[i])

        return s.value

    def ILP_KNN(self, train_features, train_labels):
        
        # number of "positive" examples
        p = sum(c==self.c[0] for c in train_labels)

        # number of "negative" examples
        n = sum(c==self.c[1] for c in train_labels)

        # KNN parameter
        k = self.csize

        # D matrix computation
        D = np.zeros((p,n))
        for i,v in enumerate(train_features[train_labels==self.c[0]]):
            for j,w in enumerate(train_features[train_labels==self.c[1]]):
                D[i,j] = np.linalg.norm(v-w)

        # Np computation
        Np = np.zeros((p,k))
        for i in range(0,p):
            Np[i,:] = np.argsort(D[i,:])[0:k]

        # Nn computation
        Nn = np.zeros((n,k))
        for j in range(0,n):
            Nn[j,:] = np.argsort(D[:,j])[0:k]

        # Wp computation
        Wp = np.zeros((p,n))
        for i in range(0,p):
            for j in range(0,n):
                Wp[i,j] = 


    def ILP_selection(self, train_features, train_labels):
        '''
        Input:
        train_features      :   numpy array of shape (N,M)  :   entire training set
        train_labels        :   numpy array of shape (N,)   :   entire training labels

        Output:
        self.train_set[]    :   numpy array of shape (n,M)  :   selected training samples
        self.train_label[]  :   numpy array of shape (n,)   :   selected training labels
        '''
        NUM_C = 200
        clf = KMeans(n_clusters=NUM_C, init="random", n_init=1, random_state=2)
        clf.fit(train_features[train_labels==self.c[0],:])
        clusters_1 = clf.predict(train_features[train_labels==self.c[0],:])
        centers_1 = clf.cluster_centers_

        clf.fit(train_features[train_labels==self.c[1],:])
        clusters_2 = clf.predict(train_features[train_labels==self.c[1],:])
        centers_2 = clf.cluster_centers_

        sorter = np.zeros((NUM_C*2,2))
        for i1,x1 in enumerate(centers_1):
            sorter[i1,0] = np.linalg.norm(centers_2-x1,axis=1).min()
            sorter[i1,1] = i1
        for i2,x2 in enumerate(centers_2):
            sorter[NUM_C+i2,0] = np.linalg.norm(centers_1-x2,axis=1).min()
            sorter[NUM_C+i2,1] = NUM_C+i2
        sorter = sorted(sorter, key = lambda x: x[0])
        sorted_cid = []
        for x in sorter:
            sorted_cid.append(x[1])
        sorted_cid = np.array(sorted_cid).astype(int)

        x = np.concatenate([centers_1,centers_2],axis=0)
        y = [0]*NUM_C + [1]*NUM_C

        # cluster id
        cid = np.zeros(np.shape(train_labels))
        cid[train_labels==self.c[0]] = clusters_1
        cid[train_labels==self.c[1]] = clusters_2 + NUM_C

        # distance opposite class
        doc = np.zeros(np.shape(train_labels))
        for i,v in enumerate(train_features):
            # doc[i] = np.linalg.norm([np.linalg.norm(centers_1-v,axis=1).min(),np.linalg.norm(centers_2-v,axis=1).min()])
            if train_labels[i] == self.c[0]:
                doc[i] = np.linalg.norm(centers_2-v,axis=1).min()
            else:
                doc[i] = np.linalg.norm(centers_1-v,axis=1).min()

        # ILP formulation
        s = cp.Variable(np.shape(train_labels)[0])
        objective = cp.Minimize(doc.T@s)
        constraints = [ cp.sum(s) >= int(np.shape(train_labels)[0] * 0.025 + 100*(1 - np.exp(-np.shape(train_labels)[0]))) ]
        for i in range(0,np.shape(train_labels)[0]):
            constraints += [
                s[i] <= 1,
                s[i] >= 0
            ]
        for i in range(0,NUM_C*2):
            pos = np.where(sorted_cid==i)[0]
            if pos<10:
                multiplier = 0.1
            elif pos<20:
                multiplier = 0.2
            elif pos<30:
                multiplier = 0.15
            else:
                multiplier = 0.1
            # multiplier = 0.1
            constraints += [
                cp.sum(s[cid==i]) <= multiplier*sum(1 for j in cid if j == i)
            ]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        for i,v in enumerate(s.value):
            if v >= 0.5:
                self.train_set=np.append(self.train_set, np.reshape(train_features[i],(1,len(train_features[i]))), axis=0)
                self.train_label=np.append(self.train_label, train_labels[i])

        return s.value

    def cluster_selection(self, train_features, train_labels, min_samples):
        '''
        Input:
        train_features      :   numpy array of shape (N,M)  :   entire training set
        train_labels        :   numpy array of shape (N,)   :   entire training labels

        Output:
        self.train_set[]    :   numpy array of shape (n,M)  :   selected training samples
        self.train_label[]  :   numpy array of shape (n,)   :   selected training labels
        '''
        n = np.shape(train_labels)[0]
        y = np.zeros(n)
        y[train_labels==self.c[0]] = 1
        y[train_labels==self.c[1]] = -1
        NUM_S = 10
        NUM_C = int(n/NUM_S)
        clf = KMeans(n_clusters=NUM_C, init="random", n_init=1, random_state=2)
        clf.fit(train_features)
        cid = clf.predict(train_features)
        cen = clf.cluster_centers_
        # self.plot_scatter(cen,np.ones(NUM_C),show=True)

        # ILP formulation
        # s = cp.Variable(NUM_C,integer=True)
        # t = cp.Variable(NUM_C)
        # constraints = []
        # constraints += [ cp.sum(s) >= min_samples/NUM_S ]
        # for i in range(0,NUM_C):
        #     constraints += [
        #         t[i] >= cp.abs(sum(y[cid==i])*s[i]),
        #         s[i] <= 1,
        #         s[i] >= 0
        #     ]
            
        # objective = cp.Minimize(cp.sum(t)-cp.sum(s))
        # prob = cp.Problem(objective, constraints)
        # prob.solve()
        # c_select = s.value

        c_select = np.zeros(NUM_C)
        for i in range(0,NUM_C):
            temp = 0
            c_select[i] = 0
            for j in y[cid==i]:
                if temp == 0:
                    temp = j
                    continue
                if j != temp:
                    c_select[i] = 1
                    break

        selected = np.zeros(n)
        for i,v in enumerate(c_select):
            if v >= 0.5:
                selected[cid==i] = 1
                self.train_set=np.append(self.train_set, train_features[cid==i], axis=0)
                self.train_label=np.append(self.train_label, train_labels[cid==i])

        return selected

    def ILP_selection_2(self, train_features, train_labels):
        n = np.shape(train_labels)[0]
        y = np.zeros(n)
        y[train_labels==self.c[0]] = 1
        y[train_labels==self.c[1]] = -1
        x = np.append(train_features,np.ones([n,1]),axis=1)
        M = 100

        # #  --------- ILP start ---------  #
        # e = cp.Variable(np.shape(x)[0])
        # t = cp.Variable(np.shape(x)[0])
        # w = cp.Variable(np.shape(x)[1])
        # objective = cp.Minimize(cp.sum(cp.abs(e-t)) + self.lamda*cp.norm(w,2))
        # constraints = []
        # for i in range(0,np.shape(x)[0]):
        #     constraints += [
        #         y[i]*(x[i].T@w) >= 1 - e[i],
        #         cp.sum(cp.abs(t)) <= 100,
        #         e[i] >= 0
        #     ]
        # prob = cp.Problem(objective, constraints)
        # prob.solve()
        # #  --------- ILP end  ---------  #

        #  --------- ILP start ---------  #
        l = cp.Variable(np.shape(x)[0])
        w = cp.Variable(np.shape(x)[1])
        objective = cp.Minimize(cp.sum(l))
        constraints = [ cp.sum(l) >=  100 ]
        for i in range(0,np.shape(x)[0]):
            constraints += [
                l[i] >= 0,
                l[i] <= 1,
                y[i]*(x[i].T@w) >= 1 - M*(1-l[i])
            ]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        #  --------- ILP end  ---------  #

        selected = np.zeros(np.shape(train_labels))
        for i,v in enumerate(l.value):
            if v >= 0.5:
                selected[i] = 1
                self.train_set=np.append(self.train_set, np.reshape(train_features[i],(1,len(train_features[i]))), axis=0)
                self.train_label=np.append(self.train_label, train_labels[i])

        return selected

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
        objective = cp.Minimize(v1.T@t/np.shape(x)[0] + self.lamda*cp.norm(a-w,2))
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