# LogisticRegression
numpy implementation of mere and locally-weighted logistic regression for binary classification problem.

---- Input
In both cases training data X,Y is passed to the class constructor.
X - 2D array of covariates, each row corresponds to one training instance.
Y - vector or 1D array of appropriate (ordered the same way rows of X are) labels.
Note:
In ordinary logistic regression it is assumed that input X does not contain column of ones responsible for bias, and is
automatically added, while input for LW logistic regression should be augmented with that column manually.  


---- logistic_regression.py contains class which implements ordinary l_2-regularized logistic regression.
The likelihood is optimized via Newton's method.


---- LWLogR.py file contains class which implements l_2-regularized locally-weighted logistic regression.
The weightening function is chosen to be Gaussian function.
The likelihood is optimized via Newton's method.


---- Usage:
  Ordinary 
Call train(reg, accuracy) to train the model. Values of parameter vector is stored in class variable theta.
Call predict(self, x, probabilities) to obtain predictions. 
reg - regularization constant (optional);
accuracy - algorithm stops when two-norm of step parameters update is equal to specified value (optional);
x - matrix of observations to predict;
probabilities - boolean parameter, set to True if you want probabilities to be output besides labels(optional).


  To predict simply call predict(x, tau, reg, accuracy) from class instance.
Here:
x - row of covariates to be labeled;
tau - weight bandwidth parameter (optional);
reg - regularization constant (optional);
accuracy - algorithm stops when two-norm of step parameters update is equal to specified value (optional).  


---- Repo also contains example of training dataset x.dat, y.dat.

---- plot_test_data.py shows how to use LWLogR.
