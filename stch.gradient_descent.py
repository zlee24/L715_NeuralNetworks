#Stochastic gradient descent 
import numpy as np
import matplotlib.pyplot as plt


num = 100
learning_rate = .01
W = np.random.rand(1)

np.random.seed(4)
X = 2 * np.random.rand(num)
Y = 4 * X + np.random.randn(num)

#plt.scatter(X,Y)


#--------------- random points plotted-----------------

def cost(X,Y,W):
    m = len(Y)

    #np.random.shuffle(X)
    yh = [np.dot(val,W) for val in X]
    comp = [(pred - real)**2 for pred, real in zip(yh,Y)]
    error = sum(comp)
    c = 1/(2*m) * error
    return c

#---------------- cost defined --------------------------

def pder(X,Y,W):
    m = len(Y)
    der = np.dot(X, np.subtract(W*X, Y))
    return der/m

#Accounted for partial derivation

def stch_gradient_descent(X, Y, W, epochs):
    m = int(len(Y))

    for i in range(epochs):

        np.random.shuffle(X)

        c = cost(X,Y,W)

        for z in range(m):
            W = W - (learning_rate * pder(X, Y, W))
        
        print(i,c)



stch_gradient_descent(X,Y,W,epochs = 20)



