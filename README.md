# 236A

## Setting up the environment
Download all the files of the repo in a folder. That's it lol. For the CSV files, click on the file from above and click 'Download' or 'View raw' and save whatever comes on display. Otherwise, the CSV won't download directly as it's too large.

## How to use test1.py
Run the below command in terminal to run the test script in 'interactive mode'
```
python -i test1.py
```
This will compile the script and open the python shell. The script defines a testing object called "test" as: `test = tester()`(no need to type this, it's included in the script).

Additional command line flags:
```
python -i test1.py --train train_file_name.csv --test test_file_name.csv
```
- `train_file_name.csv` : name of the training dataset file, default = 'mnist_train.csv'
- `test_file_name.csv` : name of the testing dataset file, default = 'mnist_test.csv'

Then you can run the classification task by typing the below command:
```
test.run()
```
This "run" method can take the  below arguments:
- `class1` : first label from all set of labels to be considered for classification, default = 1
- `class2` : second label from all set of labels to be considered for classification, default = 7
- `normal` : bool value(0 or 1) to specify whether to normalize the input data, default = 0
- `lamda` : regularization parameter, default = 0.1
- `select` : whether to use sample_selection, default = 1
- `prob` : the probability of selecting each sample in case of random sample selection, default = 0
- `init` : the number of initial number of unfiltered samples selected by the sample selector, default = 2
- `skip` : # of times to skip re-training after sample selection, default = 0
- `stop` : If train_error on remaining train_data less than this value, stops sample_selection, default = 0

Example, to run classification between digits 2 and 5 with normalization of the data:
```
test.run(class1=2,class2=5,normal=1)
```
![Example Screenshot](https://i.ibb.co/7t8shsQ/Screenshot-2021-11-13-at-3-51-47-PM.png)

## Miscellaneous
- `'lp'` is the Linear Programming method for getting the best hyperplane dividing the classes. When sample selection is used, then we adaptively reformulate the LP in such a way that the current point is correctly classified and the new hyperplan is as "close" in l-1 sense to the prev hyperplane as possible.
- `normal` is the flag for normalising the data. It showed improvements by giving at most 1% of lesser error % (ex 3% errors reduced to 2%). So made sense to use this. But I've implemented this using scipy library, need to use similar API from cvxpy or make it ourselves maybe.
- `skip` is as explained in prev section. So I observed that by doing this, we can reduce the sample selection processing time by a lot while having a trade off with slightly lesser # of samples selected and slightly more test error %. But it's not that bad an idea either I think, need to think more about this.

## Synthetic Dataset
To load the synthetic dataset instead of MNIST, run below command (500 here is size of test set, 4 times of that will be train set size):
```
python -i test1.py -s 500
```
Inside the script, use the same commands as before to run the classifier. Just make sure not to use "class1" and "class2" flags in this case. 

Additional command line flag:
```
python -i test1.py -s 500 -p
```
- `-p`: This flag will show a live plot during the sample selection as shown below:
![Live Plot](https://media2.giphy.com/media/ZEBt2RxYQYQxVKE2RL/giphy.gif)

Also, after you train using "test.run()" you can plot the test set data and the selected test set data by using below command:
```
test.plot()
```
You will get plot like this:
![Selected Data Plot](https://i.ibb.co/dWpG8gH/Whats-App-Image-2021-11-14-at-3-42-16-AM.jpg)

## References:
Sample Selection and GD Inspiration: http://image.diku.dk/jank/papers/WIREs2014.pdf  
GD python code for hinge loss: https://towardsdatascience.com/svm-implementation-from-scratch-python-2db2fc52e5c2  
Skipping re-training every sample (sec 3.2): http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.31.6090&rep=rep1&type=pdf  
Avoiding selecting "similar" samples, adding "diversity": https://www.aaai.org/Papers/ICML/2003/ICML03-011.pdf  
SVM using CVXPY: https://www.cvxpy.org/examples/machine_learning/svm.html   
Adaptive reformulation of LP at every input occurence: https://www.cs.cmu.edu/~juny/Prof/papers/kdm07jyang.pdf   
