
"""creating a basic neuron with 3 input and two ouput"""
#importing libraries
import numpy as np
activation = np.tanh #activation function we can use it as sigma tanh or linear as per out reqirements

x = np.array([0.3, 0.4, 0.1]) #Three input

W = np.array([[-2, 4, -1],[6, 0, -3]]) #Weight matrix
b = np.array([0.1, -2.5]) #Bias Matrix

n=np.dot(W,x) #Dot produt of both the vector

a1 = activation(np.dot(W,x)+b.T) # We are adding bias and the passing the result to the activation function
print(a1)