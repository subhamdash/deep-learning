"A simple example of back propogation"
import numpy as np
# Let's start with an unfit weight and bias.
w1 = 2.3
b1 = -1.2
# We can test on a single data point pair of x and y.
x = 0
y = 1

 
def initial(w1,b1,x,y):
    z=w1*x+b1 
    calculated_out=np.tanh(z)
    cost_func=(calculated_out-y)**2
    print(cost_func)
    
    dCda=2*(calculated_out-y)      #dCost/dActivation
    dadz=1/np.cosh(z)**2           #dActivation/dz
    
    dzdb=1                         #dZ/dBias
    dzdw=x                         #dz/dWeight
    
    #dCost/dBias=(dCost/dActivation)*(dActivation/dz)*(dZ/dBias)
    updated_bias=dCda*dadz*dzdb  
    
    #dCost/dWeight=(dCost/dActivation)*(dActivation/dz)*(dZ/dWeight)
    updated_weight=dCda*dadz*dzdw
    
    #Parameters is a list of both
    parameters=[updated_bias,updated_weight]
    return parameters

print(initial(w1,b1,x,y))
    
    