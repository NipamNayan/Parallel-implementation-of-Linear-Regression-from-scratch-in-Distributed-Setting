import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.size
from time import time


def Stochastic_Gradient_Descent(x, y, alpha, B):
    x = x * 1.0
    y = y * 1.0
    for i in range(0,len(x)):
        
        y_prediction = np.dot(B, x[i].T)     
        error = y_prediction - y[i]
        gradient = 2*(error)*B
        learning_rate =alpha
        
        B = B - ((learning_rate * gradient))
        
    return B


def cost_function(X, Y, B):
    m = len(Y)
    J = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
    return J




data_samp= pd.read_csv("KDD_final.csv")

Start=MPI.Wtime()                         #CALCULATING START TIME AFTER DATA IS LOADED

ind = int(data_samp.shape[0] * .7)
train_X = data_samp[1:ind].drop(['TARGET_B', 'TARGET_D'], axis = 1)
train_y = data_samp.TARGET_D[1:ind]

test_X = data_samp[(ind + 1):-1].drop(['TARGET_B', 'TARGET_D'], axis = 1)
test_y = data_samp.TARGET_D[(ind + 1):-1]

x_i = np.array(train_X)
y_i = np.array(train_y)
length =int(len(x_i))
step = int(length//size)
assert len(x_i)==len(y_i)
Loss=[]

alpha = 0.0000005
max_itr= 150
np.random.seed(10)
time_epoch=[]
B1 = np.random.randint(1,2,size=((train_X.shape[1])))     

for i in range(max_itr):
     B1 = comm.bcast(B1,root=0) 
     Local_X = x_i[(rank)*step:(rank+1)*step]     
     Local_Y = y_i[(rank)*step:(rank+1)*step]
     W_collect=[]
     W = Stochastic_Gradient_Descent(Local_X, Local_Y, alpha, B1)   
     W_collect.append(W)   

     weight = comm.gather(W_collect,root=0 )   
     if rank ==0:
         B = np.mean(weight,axis = 0)  
         LOSS = cost_function(test_X,test_y, B[0])
         Loss.append(LOSS)
         Cost = Loss
         B1 = B[0] 
         B2 = B1
         End=MPI.Wtime()



try:
    print("Loss for",i,"number of iterations is:",Cost)
    print("\nWeight Matrix is",B2)
    Time= End-Start
    print("\nTime taken",Time,"secs for number of iterations:",i)
    #plt.plot(Cost)
    #plt.title('RMSE VS EPOCHS')
    #plt.xlabel('Number of Epochs')
    #plt.ylabel('Loss Calculated as RMSE')
    #plt.show()
except NameError:
    pass