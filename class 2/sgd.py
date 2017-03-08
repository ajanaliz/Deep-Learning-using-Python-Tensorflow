# In this file I compare the progression of the cost function vs. iteration
# for 3 cases:
# 1) full gradient descent
# 2) batch gradient descent
# 3) stochastic gradient descent
#
# I use the PCA-transformed data to keep the dimensionality down (D=300)
# I've tailored this example so that the training time for each is feasible.
# So what I am really comparing is how quickly each type of GD can converge,
# (but not actually waiting for convergence) and what the cost looks like at
# each iteration.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime

from util import get_transformed_data, forward, error_rate, cost, gradW, gradb, y2indicator


def main():
    # get PCA transformed data
    X, Y, _, _ = get_transformed_data()
    X = X[:, :300] # the first 300 features

    # normalize X first
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mu) / std

    print "Performing logistic regression..."
    Xtrain = X[:-1000,]
    Ytrain = Y[:-1000]
    Xtest  = X[-1000:,]
    Ytest  = Y[-1000:]

    N, D = Xtrain.shape
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    # 1. full
    W = np.random.randn(D, 10) / 28 # we're setting our initial weights to be pretty small, proportional to the square root of the dimensionality
    b = np.zeros(10)
    LL = []
    lr = 0.0001
    reg = 0.01
    t0 = datetime.now()
    for i in xrange(200):
        p_y = forward(Xtrain, W, b)

        W += lr*(gradW(Ytrain_ind, p_y, Xtrain) - reg*W)
        b += lr*(gradb(Ytrain_ind, p_y) - reg*b)
        

        # do a forward pass on the test set so that we can calculate the cost on the test set and then plot that
        p_y_test = forward(Xtest, W, b)
        ll = cost(p_y_test, Ytest_ind)
        LL.append(ll)
        if i % 10 == 0: # calculate the error rate on every 10 iterations
            err = error_rate(p_y_test, Ytest)
            print "Cost at iteration %d: %.6f" % (i, ll)
            print "Error rate:", err
    p_y = forward(Xtest, W, b)
    print "Final error rate:", error_rate(p_y, Ytest)
    print "Elapsted time for full GD:", datetime.now() - t0


    # 2. stochastic
    W = np.random.randn(D, 10) / 28
    b = np.zeros(10)
    LL_stochastic = []
    lr = 0.0001
    reg = 0.01

    t0 = datetime.now()
    for i in xrange(1): # takes very long since we're computing cost for 41k samples
        # on each pass, we typically want to shuffle through the training data and the labels
        tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
        # we're actually only going to go through 500 samples because its slow
        for n in xrange(min(N, 500)): # shortcut so it won't take so long...
            # reshape x into a 2 dimensional matrix
            x = tmpX[n,:].reshape(1,D)
            y = tmpY[n,:].reshape(1,10)
            # forward pass to get the output
            p_y = forward(x, W, b)

            W += lr*(gradW(y, p_y, x) - reg*W)
            b += lr*(gradb(y, p_y) - reg*b)

            p_y_test = forward(Xtest, W, b)
            ll = cost(p_y_test, Ytest_ind)
            LL_stochastic.append(ll)

            if n % (N/2) == 0: # calculate the error rate once for every N/2 samples
                err = error_rate(p_y_test, Ytest)
                print "Cost at iteration %d: %.6f" % (i, ll)
                print "Error rate:", err
    p_y = forward(Xtest, W, b)
    print "Final error rate:", error_rate(p_y, Ytest)
    print "Elapsted time for SGD:", datetime.now() - t0


    # 3. batch
    W = np.random.randn(D, 10) / 28
    b = np.zeros(10)
    LL_batch = []
    lr = 0.0001
    reg = 0.01
    batch_sz = 500
    n_batches = N / batch_sz

    t0 = datetime.now()
    for i in xrange(50):
        tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
        for j in xrange(n_batches):
            # get the current batches input and targets
            x = tmpX[j*batch_sz:(j*batch_sz + batch_sz),:]
            y = tmpY[j*batch_sz:(j*batch_sz + batch_sz),:]
            # forward pass to get the output predictions
            p_y = forward(x, W, b)

            # Gradient descent
            W += lr*(gradW(y, p_y, x) - reg*W)
            b += lr*(gradb(y, p_y) - reg*b)

            p_y_test = forward(Xtest, W, b)
            ll = cost(p_y_test, Ytest_ind)
            LL_batch.append(ll)
            if j % (n_batches/2) == 0: # print error rate at every (number of batches)/2 iterations
                err = error_rate(p_y_test, Ytest)
                print "Cost at iteration %d: %.6f" % (i, ll)
                print "Error rate:", err
    p_y = forward(Xtest, W, b)
    print "Final error rate:", error_rate(p_y, Ytest)
    print "Elapsted time for batch GD:", datetime.now() - t0



    x1 = np.linspace(0, 1, len(LL))
    plt.plot(x1, LL, label="full")
    x2 = np.linspace(0, 1, len(LL_stochastic))
    plt.plot(x2, LL_stochastic, label="stochastic")
    x3 = np.linspace(0, 1, len(LL_batch))
    plt.plot(x3, LL_batch, label="batch")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()