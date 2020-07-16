import pandas as pd
import logging
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import ipdb   #for breakpoints
import os

##############
#### PROBLEM :
# Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new
# outlet. The chain already has trucks in various cities and you have data for
# profits and populations from the cities.


########################################
#### The square loss function
def compute_square_loss(X, y, theta):
    """
    Given a set of X, y, theta, compute the square loss for predicting y with X*theta

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D array of size (num_features)

    Returns:
        loss - the square loss, scalar
    """
    loss = 0 #initialize the square_loss

    htheta=np.dot(X,theta)    #product between X and theta
    delta= y-htheta           # difference between the labels and the hypothesis

    inner = np.dot(delta.T,delta)
    loss=np.sum(inner) / (2 * len(X))
    
    return loss


########################################
### compute the gradient of square loss function
def compute_square_loss_gradient(X, y, theta):
    """
    Compute gradient of the square loss (as defined in compute_square_loss), at the point theta.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    grad= np.zeros(theta.shape)     #intialise the gradient

    htheta=np.dot(X,theta)    #product between X and theta
    delta= htheta-y           # difference between the labels and the hypothesis
    prod=np.dot(delta.T,X)    
    grad=prod.T/X.shape[0]


    return grad


def main():
    #initialise parameter for gradient descent
    alpha=0.01

    print("Current Working Directory " , os.getcwd())
    path=os.getcwd() + "\\"
    
    #Loading the dataset
    print('loading the dataset')

    df = pd.read_csv(path +'ex1data1.txt', delimiter=',',header=None)
    X_train = df.values[:,:-1]    # data
    y_train = df.values[:,-1]     # labels

    print("Add bias term")
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # Add bias term
    y_train=y_train.reshape((y_train.shape[0], 1))

    ###########################################################
    ###### Now that we are all ready, let's calculate the cost function and the gradient
    # THE COST FUNCTION is here defined as the norm 2, which correspond to the sum of the squared values of the 
    # difference between the predicted label "X*theta" and the true label "y" 
    ###########################################################
    ### Cost function check
    ###########################################################
    theta=np.zeros((X_train.shape[1],1))
    print("Calculate the square loss")
    loss=compute_square_loss(X_train, y_train, theta)
    print("Predicted loss 32,07 , Calculated loss=",loss)

    theta[0,0]=2
    theta[1,0]=-1
    loss=compute_square_loss(X_train, y_train, theta)
    print("predicted loss 54,24 , Calculated loss=",loss)



    ###########################################################
    ### GRADIENT CHECK
    ###########################################################
    print("Initialisation of theta matrix")
    theta=np.zeros((X_train.shape[1],1))
    print("Calculate the gradient for Theta")
    grad=compute_square_loss_gradient(X_train, y_train, theta)
    print("theta pred. = -0.65, -0.05 ; theta =", alpha*grad)
 
 
    ###########################################################
    ### GRADIENT DESCENT LOOP
    ###########################################################
    niter=1500
    Loss=np.zeros((niter,1))
    ##for loop
    for iter in range(niter):
        theta0=theta
        grad0=grad

        loss=compute_square_loss(X_train, y_train, theta)
        grad=compute_square_loss_gradient(X_train, y_train, theta)
        theta=theta-alpha*grad

        Loss[iter]=loss

    ###########################################################
    ### plot the cost function with respect to the number of iteration
    x = np.linspace(1, niter,niter)
    f1=plt.figure(1)
    plt.subplot(121)
    plt.plot(x,Loss)
    plt.xlabel('iteration')
    plt.ylabel('Cost Function')

    ###########################################################
    ### PLOT THE REGRESSION CURVE
    plt.subplot(122)
    plt.plot(X_train[:,0],y_train ,'o',label="data")
    plt.plot(X_train[:,0],np.dot(X_train,theta) ,'o--',label="Regression curve")
    plt.xlabel('Population')
    plt.ylabel('Profits')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.legend(loc='lower right')
    plt.show() 


if __name__ == "__main__":
    main()
