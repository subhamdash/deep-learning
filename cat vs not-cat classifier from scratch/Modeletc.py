import numpy as np
import initialize_parameters

import L_forward
import L_backward
import matplotlib.pyplot as plt


def layer_sizes(X,Y):
    n_input=X.shape[0]
    n_output=Y.shape[0]
    return n_input,n_output


def compute_cost(AL, Y):
    
    m = Y.shape[1]
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))    
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return cost
    

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters[
            "W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters[
            "b" + str(l)] - learning_rate * grads["db" + str(l)]
    return parameters

    
def model(X,Y,num_iter=1000,learning_rate = 0.0075,print_cost=True):
    
    n_i,n_o=layer_sizes(X,Y)
    print(n_i,n_o)
    costs=[]

    architecture=[n_i, 20, 7, 5, n_o]
    parameters=initialize_parameters.initialize_parameters_deep(architecture)
    
    for i in range(0, num_iter):
        AL, caches = L_forward.L_model_forward(X, parameters)
        #print(len(AL),len(Y))
        
        cost = compute_cost(AL, Y)
        grads = L_backward.L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        #print(parameters)
        if print_cost and (i+1) % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters