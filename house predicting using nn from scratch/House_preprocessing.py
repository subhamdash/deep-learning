import numpy as np
import preprocess
import Modeletc
import predict_value



X_train, X_test, Y_train, Y_test =preprocess.load_data()


print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


train_x_flatten = X_train.reshape(X_train.shape[0], -1).T
test_x_flatten=X_test.reshape(X_test.shape[0], -1).T

train_y_flatten = Y_train.reshape(Y_train.shape[0], -1).T
test_y_flatten=Y_test.reshape(Y_test.shape[0], -1).T

print(train_x_flatten.shape,test_x_flatten.shape,train_y_flatten.shape)





def layer_sizes(X,Y):
    n_input=X.shape[0]
    n_output=Y.shape[0]
    return n_input,n_output




n_i,n_o=layer_sizes(train_x_flatten,train_y_flatten)
print(n_i,n_o)
architecture=[n_i, 600,340, 150,68, 7, n_o]


# model(X,Y,architecture,num_iter=1000,learning_rate = 0.0075,print_cost=True)
parameters = Modeletc.model(train_x_flatten, train_y_flatten,architecture,num_iter=5000,learning_rate=0.006)


#print(parameters)
pred_train = predict_value.predict(train_x_flatten, train_y_flatten, parameters)
pred_test = predict_value.predict(test_x_flatten, test_y_flatten, parameters)




