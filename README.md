# 236A

## Setting up the environment
Download all the files of the repo in a folder. That's it lol.

## How to use test1.py
Run the below command in terminal to run the test script in 'interactive mode'
```
python -i test1.py
```
This will compile the script and open the python shell. The script defines a testing object called "test" as shown below (no need to type this, it's included in the script):
```
test = tester()
```
Then you can run the classification task by typing the below command:
```
test.run()
```
This "run" method can take the  below arguments:
- `train_file` : name of the training dataset file, default = 'mnist_train.csv'
- `test_file` : name of the testing dataset file, default = 'mnist_test.csv'
- `class1` : first label from all set of labels to be considered for classification, default = 1
- `class2` : second label from all set of labels to be considered for classification, default = 7
- `normal` : bool value(0 or 1) to specify whether to normalize the input data, default = 0
- `alpha` : learning rate of gradient descent, default = 0.1
- `lamda` : regularization parameter, default = 0.1
- `epochs` : number of epochs to run, default = 50
- `method` : whether to use 'gd' or 'sgd' or 'bgd', default = 'gd'
- `select` : whether to use sample_selection, default = 0
- `shuffle` : whether to shuffle data at every epoch (only used in 'sgd'), default = 0
- `prob` : the probability of selecting each sample in case of random sample selection, use 0 for our designed sample selection, default = 0.2
- `init` : the number of initial number of unfiltered samples that are to be selected by the sample selector, default = 10
- `delta` : the range within which we'd prefer selecting the samples, distance from current hyperplane, proportional to # of samples selected, default = 1
- `skip` : # of samples to skip in between requesting for re-trained weights from main node, so will use old-weights for these many samples, default = 0

Example, to run classification between digits 2 and 5 with normalization of the data and 70 epochs:
```
test.run(class1=2,class2=5,normal=1,epochs=70)
```
![Example Screenshot](https://i.ibb.co/7t8shsQ/Screenshot-2021-11-13-at-3-51-47-PM.png)

## Miscellaneous
- `'gd'` is the basic gradient descent classifier. This is the default classifier if you don't mention anything. And by default runs 50 epochs. This is the classifier used at the end of sample_selection with 70 epochs irrespective of what setting you mentioned in the arguments. Whatever you mention in the arguments is used for intermediate training only during the sample_selection process. At the end, 'gd' is the way to go with 70 epochs - don't ask me why.
- `'sgd'` is the basic stochastic gradient descent classifier. If using this, you won't find any improvement in using more than 1 epochs unless you turn on 'shuffle' flag. Shuffle basically shuffles the training data in every epoch. Tbh even after turning on the 'shuffle' and using multiple epochs, I didn't find much improvement. Not gonna use this for now until some other approach makes this useful.
- `'bgd'` is the batch gradient descent. Tbh I have no idea what a batch GD is, I just named it that because I'd heard it somewhere and it intuitively seemed similar to what I wanted to design. So basically, I implemented this based on the idea of just using the past 10 or 20 samples because otherwise GD takes a lot of time to compute the gradient using all of the existing samples - which is a problem in sample selection because we want training to be done after each sample to determine the importance of the next.
- `normal` is the flag for normalising the data. It showed improvements by giving at most 1% of lesser error % (ex 3% errors reduced to 2%). So made sense to use this. But I've implemented this using scipy library, need to use similar API from cvxpy or make it ourselves maybe.
- `skip` is as explained in prev section. So I observed that by doing this, we can reduce the sample selection processing time by a lot while having a trade off with slightly lesser # of samples selected and slightly more test error %. But it's not that bad an idea either I think, need to think more about this.
- Other parameters: I have stuck to fixed values of learning rate and regularization parameter. Tried playing around with it a bit, didn't find much success out of the default of 0.1 for both. We can focus on this if all else is done. (ex: having a dynamic learning rate etc...)

## Synthetic Dataset
To load the synthetic dataset instead of MNIST, run below command (500 here is size of test set, 4 times of that will be train set size):
```
python -i test1.py -s 500
```
Inside the script, use the same commands as before to run the classifier. Just make sure not to use "class1" and "class2" flags in this case. 

Also, after you train using "test.run()" you can plot the test set data and the selected test set data by using below command:
```
test.plot()
```
You will get plot like this:
![Selected Data Plot](https://i.ibb.co/dWpG8gH/Whats-App-Image-2021-11-14-at-3-42-16-AM.jpg)

## References:
Sample Selection and GD Inspiration: http://image.diku.dk/jank/papers/WIREs2014.pdf  
GD python code for hinge loss: https://towardsdatascience.com/svm-implementation-from-scratch-python-2db2fc52e5c2