# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 00:51:29 2019

@author: subham
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
from pandas import get_dummies
from sklearn.model_selection import train_test_split

#read the iris dataset present in the same folder as the code
data=pd.read_csv('iris.csv') 
#if the dataset doesnt contains any columns then add it using the below command
data.columns = ['sepal_length','sepal_width','petal_length','petal_width','class']
#pairplot the dataet with all the features
g=sns.pairplot(data,hue='class',height=3)
#total number of columns in the dataset
cols=data.columns
#divide features and classes/labels
features=cols[0:4]
labels=cols[4]
#Well conditioned data will have zero mean and equal variance
#We get this automattically when we calculate the Z Scores for the data
data_norm = pd.DataFrame(data)

for feature in features:
    data[feature] = (data[feature] - data[feature].mean())/data[feature].std()

#Show that should now have zero mean
print("Averages")
print(data.mean())

print("\n Deviations")
#Show that we have equal variance
print(pow(data.std(),2))

#Shuffle The data
indices = data_norm.index.tolist()
indices = np.array(indices)
np.random.shuffle(indices)
X = data_norm.reindex(indices)[features]
y = data_norm.reindex(indices)[labels]

# One Hot Encode as a dataframe
y = get_dummies(y)

# Generate Training and Validation Sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.3)

# Convert to np arrays so that we can use with TensorFlow
train_x = np.array(X_train).astype(np.float32)
test_x  = np.array(X_test).astype(np.float32)
train_y = np.array(y_train).astype(np.float32)
test_y  = np.array(y_test).astype(np.float32)

input_neuron=4      #as our data contains only four features length and width of sepal and peta
hidden_neuron1=10   #it can be depent on the neural network hidden layers 
hidden_neuron2=5    
output_class=3      #the total classes or labels present in datastet

X = tf.placeholder(tf.float32, shape=[None, input_neuron])
y = tf.placeholder(tf.float32, shape=[None, output_class])
#dictionary for the weights and biases
weights = {
    'h1': tf.Variable(tf.random_normal([input_neuron, hidden_neuron1])),
    'h2': tf.Variable(tf.random_normal([hidden_neuron1, hidden_neuron2])),
    'out': tf.Variable(tf.random_normal([hidden_neuron2, output_class]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([ hidden_neuron1])),
    'b2': tf.Variable(tf.random_normal([ hidden_neuron2])),
    'out': tf.Variable(tf.random_normal([output_class]))
}
#layers and activation layer for the neural network
layer_1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
layer_1 = tf.nn.tanh(layer_1)
layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
layer_2 = tf.nn.tanh(layer_2)
out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
y_predict = tf.argmax(out_layer, axis=1)


learn_rate = 0.01
max_epochs = 200

cee = tf.nn.softmax_cross_entropy_with_logits_v2\
(labels=y, logits=out_layer)


cost = tf.reduce_mean(cee)


optimizer = tf.train.GradientDescentOptimizer(learn_rate)

trainer = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print("Starting training")
for epoch in range(max_epochs):
    indices = np.arange(len(train_x))
    for ii in range(len(indices)):
        i = indices[ii]
        sess.run(trainer, feed_dict={X: train_x[i:i+1],y: train_y[i:i+1]})
    train_acc = np.mean(np.argmax(train_y, axis=1) == \
sess.run(y_predict, feed_dict={X:train_x, y:train_y}))
    if epoch > 0 and epoch % 10 == 0:
        print("epoch = %4d, train accuracy = %.4f " % \
(epoch, train_acc))
print("Training complete \n")
test_acc = np.mean(np.argmax(test_y, axis=1) == \
sess.run(y_predict, feed_dict={X:test_x, y:test_y}))
print("Accuracy on test data = %.4f " % test_acc)
sess.close()





