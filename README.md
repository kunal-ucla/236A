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

Example, to run classification between digits 2 and 5 with normalization of the data and 70 epochs:
```
test.run(class1=2,class2=5,normal=1,epochs=70)
```
![Example Screenshot](https://i.ibb.co/7t8shsQ/Screenshot-2021-11-13-at-3-51-47-PM.png)

## References:
Sample Selection and GD Inspiration: http://image.diku.dk/jank/papers/WIREs2014.pdf  
GD python code for hinge loss: https://towardsdatascience.com/svm-implementation-from-scratch-python-2db2fc52e5c2