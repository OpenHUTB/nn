#!/usr/bin/env python
# coding: utf-8

# ## 准备数据

# In[1]:

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

def mnist_dataset():
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    #normalize
    x = x/255.0
    x_test = x_test/255.0
    
    return (x, y), (x_test, y_test)


# ## Demo numpy based auto differentiation

# In[3]:


import numpy as np

class Matmul:
    def __init__(self):
        self.mem = {}
        
    def forward(self, x, W):
        h = np.matmul(x, W)
        self.mem={'x': x, 'W':W}
        return h
    
    def backward(self, grad_y):
        '''
        x: shape(N, d)
        w: shape(d, d')
        grad_y: shape(N, d')
        '''
        x = self.mem['x']
        W = self.mem['W']
        
        ####################
        '''计算矩阵乘法的对应的梯度'''
        ####################
        return grad_x, grad_W


class Relu:
    def __init__(self):
        self.mem = {}
        
    def forward(self, x):
        self.mem['x']=x
        return np.where(x > 0, x, np.zeros_like(x))
    
    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''
        ####################
        '''计算relu 激活函数对应的梯度'''
        ####################
        return grad_x

class Softmax:
    '''
    softmax over last dimention
    '''
    def __init__(self):
        self.epsilon = 1e-12
        self.mem = {}
        
    def forward(self, x):
        '''
        x: shape(N, c)
        '''
        x_exp = np.exp(x)
        partition = np.sum(x_exp, axis=1, keepdims=True)
        out = x_exp/(partition+self.epsilon)
        
        self.mem['out'] = out
        self.mem['x_exp'] = x_exp
        return out
    
    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''
        s = self.mem['out']
        sisj = np.matmul(np.expand_dims(s,axis=2), np.expand_dims(s, axis=1)) # (N, c, c)
        g_y_exp = np.expand_dims(grad_y, axis=1)
        tmp = np.matmul(g_y_exp, sisj) #(N, 1, c)
        tmp = np.squeeze(tmp, axis=1)
        tmp = -tmp + grad_y * s 
        return tmp
    
class Log:
    '''
    softmax over last dimention
    '''
    def __init__(self):
        self.epsilon = 1e-12
        self.mem = {}
        
    def forward(self, x):
        '''
        x: shape(N, c)
        '''
        out = np.log(x+self.epsilon)
        
        self.mem['x'] = x
        return out
    
    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''
        x = self.mem['x']
        
        return 1./(x+1e-12) * grad_y

# ## Gradient check

# In[5]:


# import tensorflow as tf

# x = np.random.normal(size=[5, 6])
# W = np.random.normal(size=[6, 4])
# aa = Matmul()
# out = aa.forward(x, W) # shape(5, 4)
# grad = aa.backward(np.ones_like(out))
# print (grad)

# with tf.GradientTape() as tape:
#     x, W = tf.constant(x), tf.constant(W)
#     tape.watch(x)
#     y = tf.matmul(x, W)
#     loss = tf.reduce_sum(y)
#     grads = tape.gradient(loss, x)
#     print (grads)

# import tensorflow as tf

# x = np.random.normal(size=[5, 6])
# aa = Relu()
# out = aa.forward(x) # shape(5, 4)
# grad = aa.backward(np.ones_like(out))
# print (grad)

# with tf.GradientTape() as tape:
#     x= tf.constant(x)
#     tape.watch(x)
#     y = tf.nn.relu(x)
#     loss = tf.reduce_sum(y)
#     grads = tape.gradient(loss, x)
#     print (grads)

# import tensorflow as tf
# x = np.random.normal(size=[5, 6], scale=5.0, loc=1)
# label = np.zeros_like(x)
# label[0, 1]=1.
# label[1, 0]=1
# label[1, 1]=1
# label[2, 3]=1
# label[3, 5]=1
# label[4, 0]=1
# print(label)
# aa = Softmax()
# out = aa.forward(x) # shape(5, 6)
# grad = aa.backward(label)
# print (grad)

# with tf.GradientTape() as tape:
#     x= tf.constant(x)
#     tape.watch(x)
#     y = tf.nn.softmax(x)
#     loss = tf.reduce_sum(y*label)
#     grads = tape.gradient(loss, x)
#     print (grads)

# import tensorflow as tf

# x = np.random.normal(size=[5, 6])
# aa = Log()
# out = aa.forward(x) # shape(5, 4)
# grad = aa.backward(label)
# print (grad)

# with tf.GradientTape() as tape:
#     x= tf.constant(x)
#     tape.watch(x)
#     y = tf.math.log(x)
#     loss = tf.reduce_sum(y*label)
#     grads = tape.gradient(loss, x)
#     print (grads)


# # Final Gradient Check

# In[6]:

import tensorflow as tf

label = np.zeros_like(x)
label[0, 1]=1.
label[1, 0]=1
label[2, 3]=1
label[3, 5]=1
label[4, 0]=1

x = np.random.normal(size=[5, 6])
W1 = np.random.normal(size=[6, 5])
W2 = np.random.normal(size=[5, 6])

mul_h1 = Matmul()
mul_h2 = Matmul()
relu = Relu()
softmax = Softmax()
log = Log()

h1 = mul_h1.forward(x, W1) # shape(5, 4)
h1_relu = relu.forward(h1)
h2 = mul_h2.forward(h1_relu, W2)
h2_soft = softmax.forward(h2)
h2_log = log.forward(h2_soft)


h2_log_grad = log.backward(label)
h2_soft_grad = softmax.backward(h2_log_grad)
h2_grad, W2_grad = mul_h2.backward(h2_soft_grad)
h1_relu_grad = relu.backward(h2_grad)
h1_grad, W1_grad = mul_h1.backward(h1_relu_grad)

print(h2_log_grad)
print('--'*20)
# print(W2_grad)

with tf.GradientTape() as tape:
    x, W1, W2, label = tf.constant(x), tf.constant(W1), tf.constant(W2), tf.constant(label)
    tape.watch(W1)
    tape.watch(W2)
    h1 = tf.matmul(x, W1)
    h1_relu = tf.nn.relu(h1)
    h2 = tf.matmul(h1_relu, W2)
    prob = tf.nn.softmax(h2)
    log_prob = tf.math.log(prob)
    loss = tf.reduce_sum(label * log_prob)
    grads = tape.gradient(loss, [prob])
    print (grads[0].numpy())


# ## 建立模型

# In[10]:


class myModel:
    def __init__(self):
        
        self.W1 = np.random.normal(size=[28*28+1, 100])
        self.W2 = np.random.normal(size=[100, 10])
        
        self.mul_h1 = Matmul()
        self.mul_h2 = Matmul()
        self.relu = Relu()
        self.softmax = Softmax()
        self.log = Log()
        
        
    def forward(self, x):
        x = x.reshape(-1, 28*28)
        bias = np.ones(shape=[x.shape[0], 1])
        x = np.concatenate([x, bias], axis=1)
        
        self.h1 = self.mul_h1.forward(x, self.W1) # shape(5, 4)
        self.h1_relu = self.relu.forward(self.h1)
        self.h2 = self.mul_h2.forward(self.h1_relu, self.W2)
        self.h2_soft = self.softmax.forward(self.h2)
        self.h2_log = self.log.forward(self.h2_soft)
            
    def backward(self, label):
        self.h2_log_grad = self.log.backward(-label)
        self.h2_soft_grad = self.softmax.backward(self.h2_log_grad)
        self.h2_grad, self.W2_grad = self.mul_h2.backward(self.h2_soft_grad)
        self.h1_relu_grad = self.relu.backward(self.h2_grad)
        self.h1_grad, self.W1_grad = self.mul_h1.backward(self.h1_relu_grad)
        
model = myModel()


# ## 计算 loss

# In[11]:


def compute_loss(log_prob, labels):
     return np.mean(np.sum(-log_prob*labels, axis=1))
    
def compute_accuracy(log_prob, labels):
    predictions = np.argmax(log_prob, axis=1)
    truth = np.argmax(labels, axis=1)
    return np.mean(predictions==truth)

def train_one_step(model, x, y):
    model.forward(x)
    model.backward(y)
    model.W1 -= 1e-5* model.W1_grad
    model.W2 -= 1e-5* model.W2_grad
    loss = compute_loss(model.h2_log, y)
    accuracy = compute_accuracy(model.h2_log, y)
    return loss, accuracy

def test(model, x, y):
    model.forward(x)
    loss = compute_loss(model.h2_log, y)
    accuracy = compute_accuracy(model.h2_log, y)
    return loss, accuracy

# ## 实际训练

# In[12]:

def prepare_data():
    train_data, test_data = mnist_dataset()
    train_label = np.zeros(shape=[train_data[0].shape[0], 10])
    test_label = np.zeros(shape=[test_data[0].shape[0], 10])
    train_label[np.arange(train_data[0].shape[0]), np.array(train_data[1])] = 1.
    test_label[np.arange(test_data[0].shape[0]), np.array(test_data[1])] = 1.
    return train_data[0], train_label, test_data[0], test_label

def train(model, train_data, train_label, epochs=50):
    losses = []
    accuracies = []
    for epoch in tqdm(range(epochs), desc="Training"):
        loss, accuracy = train_one_step(model, train_data, train_label)
        losses.append(loss)
        accuracies.append(accuracy)
        print(f'Epoch {epoch}: Loss {loss:.4f}; Accuracy {accuracy:.4f}')
    return losses, accuracies

if __name__ == "__main__":
    train_data, train_label, test_data, test_label = prepare_data()
    model = myModel()
    losses, accuracies = train(model, train_data, train_label)
    test_loss, test_accuracy = test(model, test_data, test_label)
    print(f'Test Loss {test_loss:.4f}; Test Accuracy {test_accuracy:.4f}')    
