import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import scipy.io
import matplotlib.pyplot as plt
import pickle
import numpy.linalg 
from numpy.linalg import inv
from numpy.linalg import det
import math


def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    k=np.unique(y)
    y=y.flatten()
    means=np.zeros((X.shape[1],k.shape[0]))
    for i in range(k.size):
        classmean=X[y==k[i]]
        means[:,i]=np.mean(classmean,axis=0)
    
    covmat=np.cov(X,rowvar=0)
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    k=np.unique(y)
    y=y.flatten()
    means=np.zeros((X.shape[1],k.shape[0]))
    covmats=[np.zeros((X.shape[1],X.shape[1]))] * k.shape[0]
    for i in range(k.size):
        classmean=X[y==k[i]]
        means[:,i]=np.mean(classmean,axis=0)
        covmats[i]=np.cov(classmean,rowvar=0)   
    
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD
    inverse=inv(covmat)
    determinant=det(covmat)
    distance=np.zeros((Xtest.shape[0],means.shape[1]))
    for i in range(means.shape[1]):
        a=np.exp(-1*np.sum((Xtest - means[:,i])*np.dot((Xtest - means[:,i]),inverse),1)/2)
        b=sqrt(2*math.pi)*(determinant**2)       
        distance[:,i]=a/b
    maximum=np.argmax(distance,1)+1
    ytest=ytest.flatten()
    acc=100*np.mean(maximum==ytest)
    return acc

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD
    distance= np.zeros((Xtest.shape[0],means.shape[1]))
    for i in range(means.shape[1]):
        inverse = inv(covmats[i])
        determinant = det(covmats[i])
        a=np.exp(-1*np.sum((Xtest - means[:,i])*np.dot((Xtest - means[:,i]),inverse),1)/2)
        b=sqrt(2*math.pi)*(determinant**2)
        distance[:,i]=a/b   
    maximum= np.argmax(distance,1)+1    
    ytest = ytest.flatten()
    acc = 100*np.mean(maximum == ytest)
    return acc

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD   
    I=np.dot(X.T,X)
    inverse=inv(I)
    x_inverse=np.dot(inverse,X.T)
    w=np.dot(x_inverse,y)                                                
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD    
    a=np.identity((np.dot(X.T,X)).shape[0])
    inverse=inv((lambd*X.shape[0]*a)+np.dot(X.T,X))
    w=np.dot(np.dot(inverse,X.T),y)
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse   
    # IMPLEMENT THIS METHOD
    b=np.dot(Xtest,w)
    a=np.square(ytest-b)
    rmse=sqrt(np.sum(a))/Xtest.shape[0]  
    return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD 
    w=np.mat(w)
    w=w.T
    a=y-np.dot(X,w)
    J=np.dot(a.T,a)/(2*X.shape[0])
    l=lambd*np.dot(w.T,w)/2
    error=J+l
    
    dJ1=np.dot(w.T,np.dot(X.T,X))    
    dJ2=np.dot(y.T,X)    
    dJ3=lambd*w.T
    
    grad=((dJ1-dJ2)/X.shape[0])+dJ3
    error_grad=np.array(grad).flatten()   
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD
    Xd=numpy.zeros(shape=(x.shape[0],p+1))
    for i in range(x.shape[0]):
        for j in range(p+1):
            Xd[i][j]=np.power(x[i],j)
            
    return Xd

def plot1():        
    inverse = inv(covmat)
    determinant = det(covmat)
    distance= np.zeros((mesh2.shape[0],means.shape[1]))
    for i in range(means.shape[1]):    
        a=np.exp(-1*np.sum((mesh2 - means[:,i])*np.dot((mesh2 - means[:,i]),inverse),1)/2)
        b=sqrt(2*math.pi)*(determinant**2)
        distance[:,i]=a/b   
    maximum= np.argmax(distance,1)+1 
    for i in range (mesh2.shape[0]):        
        plt.scatter(mesh2[i,0],mesh2[i,1],c=colors[maximum[i]-1])
        
def plot2():
    distance= np.zeros((mesh.shape[0],means.shape[1]))
    for i in range(means.shape[1]):
        inverse = inv(covmats[i])
        determinant = det(covmats[i])
        a=np.exp(-1*np.sum((mesh - means[:,i])*np.dot((mesh - means[:,i]),inverse),1)/2)
        b=sqrt(2*math.pi)*(determinant**2)
        distance[:,i]=a/b   
    maximum= np.argmax(distance,1)+1    
    for i in range (mesh.shape[0]):
        plt.scatter(mesh[i,0],mesh[i,1],c=colors[maximum[i]-1])        

# Main script

# Problem 1
# load the sample data                                                                 
X,y,Xtest,ytest = pickle.load(open('/Users/saisrinath/Projects/Canopy_Projects/Programming Assignment 2/sample.pickle','rb'))            
#creating meshes for plots
a=np.linspace(0,16,num=100)
b=np.linspace(0,16,num=100)
x1, x2 = np.meshgrid(a,b)
mesh=np.vstack((x1.reshape(x1.size),x2.reshape(x2.size))).T
mesh2=np.vstack((x1.reshape(x1.size),x2.reshape(x2.size))).T
colors=['b','c','r','m','y'] 
# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
#plotting for LDA
plt.figure()
plot1()
plt.show()          
# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))
#plotting for QDA
plt.figure()
plot2()
plt.show()
# Problem 2

X,y,Xtest,ytest = pickle.load(open('/Users/saisrinath/Projects/Canopy_Projects/Programming Assignment 2/diabetes.pickle','rb'))   
# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses3 = np.zeros((k,1))
rmses3_train = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    rmses3_train[i] = testOLERegression(w_l,X_i,y)
    i = i + 1
plt.figure()
plt.plot(lambdas,rmses3)
plt.plot(lambdas,rmses3_train)
plt.legend(('Test Data','Train Data'))
plt.show()

# Problem 4
k = 101
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses4 = np.zeros((k,1))
rmses4_train = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.zeros((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l_1 = np.zeros((X_i.shape[1],1))
    for j in range(len(w_l.x)):
        w_l_1[j] = w_l.x[j]
    rmses4[i] = testOLERegression(w_l_1,Xtest_i,ytest)
    rmses4_train[i] = testOLERegression(w_l_1,X_i,y)
    
    i = i + 1
plt.figure()
plt.plot(lambdas,rmses4)
plt.plot(lambdas,rmses4_train)
plt.legend(('Test Data','Train Data'))
plt.show()



# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
rmses5_train=np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    rmses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
    rmses5_train[p,1] = testOLERegression(w_d2,Xd,y)
plt.figure()
plt.plot(range(pmax),rmses5)
plt.plot(range(pmax),rmses5_train)
plt.legend(('Test Data:No Regularization','Test Data:Regularization','Train Data:No Regularization','Train Data:Regularization'))
plt.show()




