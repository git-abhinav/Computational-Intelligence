
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Perceptron
x = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
y = np.array([0, 1, 1, 1])
model = Perceptron()
model.fit(x, y)
print('Output: %d' % int(model.predict([[0, 0]])))


# In[ ]:




# In[14]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np

class Perceptron(object):

    def __init__(self, no_of_inputs, n=100, learning_rate=0.05):
        '''
        Objective: To initialize data memebers
        Input Parameters: s
            self - object of type Perceptron
            no_of_inputs, epochs, learning_rate - int
        Return Value: None
        '''
        self.epochs = n
        self.learning_rate = learning_rate
        self.weights = 0.05 * np.random.randn(no_of_inputs + 1)# weights corr to num_inputs + 1 for bias 
        print("Initial values of weights",self.weights)
    def predict(self, inputs):
        '''
        Objective: To predict the output
        Input Parameters: 
            self - object of type Perceptron
            inputs - numpy array
        Return Value: Binary
        '''
        summation = np.dot(self.weights[1:].T, inputs) + self.weights[0]
        if summation > 0:
          activation = 1
        else:
          activation = 0            
        return activation

    def train(self, training_inputs, labels):
        '''
        Objective: To train the model so as to tune weights
        Input Parameters:  
                self - object of type Perceptron
                training_inputs, labels - np array
        Return Value: None
        '''
        for _ in range(self.epochs):
            deltaWeights = np.array([0,0]).astype('float64')
            deltaBias = 0.0
            nInstances = training_inputs.shape[0]
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                deltaWeights += self.learning_rate * (label - prediction) * inputs
                deltaBias += self.learning_rate * (label - prediction)
            self.weights[1:] += (1/nInstances)*deltaWeights
            self.weights[0] += (1/nInstances)*deltaBias
            
            
            
            
        b, w1, w2 = self.weights[0], self.weights[1], self.weights[2]
        x = np.array([i for i in range(-1,3)])
        y = (-1)*(b+w1*x)/w2
        plt.plot(x, y)

        
                


# In[20]:


training_inputs = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
labels = np.array([0, 0, 0, 1])
for i,j in zip(training_inputs, labels):
    print(i, j)
    


# In[15]:


a = np.array([[1, 0], [0, 1]])
b = np.array([4, 1])
np.dot(a, b)


# In[21]:



training_inputs = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
labels = np.array([0, 0, 0, 1])
#labels = np.array([0, 1, 1, 1])
plt.scatter([i for i, j in training_inputs], [j for i, j in training_inputs])

perceptron = Perceptron(2)
perceptron.train(training_inputs, labels)
result = perceptron.predict(np.array([0, 1])) 
print("Output",result)

plt.show()

