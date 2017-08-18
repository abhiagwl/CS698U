import os
import numpy as np
import time 
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import signal as sg
import math
import random

class tanh(object):
    @staticmethod
    def fn(z):
        return np.tanh(z)
    @staticmethod
    def derivative(z):
        return np.power(np.cosh(z),-2)

class ReLU(object):
    @staticmethod
    def fn(z):
        return  np.maximum(z,0)
    @staticmethod
    def derivative(z):
        return 1.0*(z>0)
class Network(object):

    def __init__(self, layers, mini_batch_size):

        self.layers = layers
        self.mini_batch_size = mini_batch_size
        
   
      
    def SGD(self, training_data,validation_data, epochs, eta,name):
        
        num_training_batches = len(training_data)/self.mini_batch_size
        count =0
        valid=validation_data
        valid_acc,train_acc=[],[]
        for j in xrange(epochs):

            random.shuffle(training_data)
            
            mini_batches = [
                training_data[k:k+self.mini_batch_size]
                for k in xrange(0,len(training_data), self.mini_batch_size)]

            t=time.time()
            for mini_batch in mini_batches:
                count +=1
                self.update_mini_batch(eta,mini_batch)
                if not count%5:
                    valid_acc.append(self.accuracy(valid))
                    train_acc.append(self.accuracy(mini_batch))
            print "time for",j,"epoch", time.time()-t
            #print "accuracy: ",self.accuracy(validation_data)
            #print "cost : ", self.cost(validation_data)    
        self.plot_acc(train_acc,valid_acc,name)
            

   
    def update_mini_batch(self, eta, mini_batch):
        self.backprop(mini_batch,self.mini_batch_size)
        for i in reversed(xrange(2,len(self.layers))):
            self.layers[i].update(eta,self.mini_batch_size)
        
    
    def forwardpass(self,mini_batch,mini_batch_size):
        self.layers[0].forward(mini_batch,mini_batch_size)
        for i in xrange(1,len(self.layers)):
            self.layers[i].forward(self.layers[i-1].output,mini_batch_size)
        
    def backprop(self,mini_batch,mini_batch_size):
        self.forwardpass(mini_batch,mini_batch_size)
        self.layers[-1].gradient(mini_batch,self.layers[-2].z,mini_batch_size)
        for i in xrange(2,len(self.layers)):
            self.layers[-i].gradient(self.layers[-i+1].back_error,self.layers[-i-1].z,mini_batch_size)
        self.layers[-len(self.layers)].gradient(self.layers[-i].back_error,mini_batch,mini_batch_size)
       
    """def forwardpass(self,mini_batch,mini_batch_size):
        t=time.time()
        self.layers[0].forward(mini_batch,mini_batch_size)
        print time.time()-t, "layer",0
        for i in xrange(1,len(self.layers)):
            t=time.time()
            self.layers[i].forward(self.layers[i-1].output,mini_batch_size)
            print time.time()-t, "layer", i
        
    def backprop(self,mini_batch,mini_batch_size):
        self.forwardpass(mini_batch,mini_batch_size)
        
        t=time.time()
        self.layers[-1].gradient(mini_batch,self.layers[-2].z,mini_batch_size)
        print time.time()-t, "layer",-1
        for i in xrange(2,len(self.layers)):
            t=time.time()
            self.layers[-i].gradient(self.layers[-i+1].back_error,self.layers[-i-1].z,mini_batch_size)
            print time.time()-t, "layer",-i
        t=time.time()
        self.layers[-len(self.layers)].gradient(self.layers[-i].back_error,mini_batch,mini_batch_size)
        print time.time()-t, "layer",-len(self.layers)"""
    
    
    
    def accuracy(self,data):
        accuracy=0.0
        self.forwardpass(data,len(data))
        y_pred=[np.argmax(o) for o in self.layers[-1].output]
        y=[np.argmax(data_[1]) for data_ in data]
        accuracy=accuracy+sum(int(y1==y2) for y1,y2 in zip(y_pred,y))
        return accuracy/len(data)
    def cost(self,data):
        
        cost=0.0
        self.forwardpass(data,len(data))
        for i in xrange(len(data)):
            y_pred=self.layers[-1].output[i]
            y=data[i][1]
            cost+=np.sum(np.nan_to_num(-y*np.log(y_pred)))
        return cost/len(data)
    
    def plot_acc(self, training_acc,validation_acc,name):
        ctr = range(1,len(training_acc)+1)
        plt.plot(ctr, training_acc)
        plt.plot(ctr, validation_acc)
        plt.legend(['Training Accuracy', 'Validation Accuracy'])
        plt.ylabel('Accuracy')
        plt.xlabel('Iterations')
        plt.savefig(name)
        plt.close()
        
    def numerical_check(self, training_data,p,ind):

        random.shuffle(training_data)
        data=training_data[0:1]
        self.backprop(data,1)
        h=math.pow(10,p)
        estimate=np.zeros(self.layers[ind].weights.shape)
        for i,value in np.ndenumerate(self.layers[ind].weights):
            epsilon = np.zeros(self.layers[ind].weights.shape)
            epsilon[i]=h
            self.layers[ind].weights+=epsilon
            cost1=self.cost(data)
            self.layers[ind].weights=self.layers[ind].weights-2*epsilon
            cost2=self.cost(data)
            estimate[i]=(cost1-cost2)/(2*h)
            self.layers[ind].weights+=epsilon
        
        return (estimate,self.layers[ind].grad_w)

class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out,activation_func,flag=1):
        self.n_in = n_in
        self.n_out = n_out
        self.flag=flag
        self.act_func=activation_func

        self.weights = np.random.randn(n_out, n_in)*(1.0/np.sqrt(n_out+n_in))
              
        self.biases = np.full((n_out,1),1.0/np.sqrt(n_out+n_in))
             
        self.grad_b=np.zeros(self.biases.shape)
        self.grad_w=np.zeros(self.weights.shape)
        
        self.v_b=np.zeros(self.biases.shape)
        self.v_w=np.zeros(self.weights.shape)
        
        self.back_error=None
        self.output=None
        self.z=None
        self.delta=None


    def forward(self, inpt,mini_batch_size):

        self.input=inpt
        self.output=np.zeros((mini_batch_size,self.n_out,1))
        self.z=np.zeros((mini_batch_size,self.n_out,1))
        #if (self.flag==2) :
        for i in xrange(mini_batch_size):
            self.inpt = inpt[i].reshape(( self.n_in,1))
            self.z[i]=np.dot(self.weights,self.inpt) + self.biases
            self.output [i] = self.act_func.fn(self.z[i])
        
   
    def gradient(self,input_1,input_2,mini_batch_size):
        self.delta=input_1
        
        self.grad_b=np.zeros(self.biases.shape)
        self.grad_w=np.zeros(self.weights.shape)
        
        
        if (self.flag==2) :
            self.back_error=np.zeros(input_2.shape)

            for i in xrange(mini_batch_size):
                self.grad_b=self.grad_b+self.delta[i]
                self.grad_w=self.grad_w+np.dot(self.delta[i],np.transpose(self.act_func.fn(input_2[i])))
                
                self.back_error[i]=self.act_func.derivative(input_2[i])*(np.dot(np.transpose(self.weights),
                                                                            self.delta[i]))
        elif (self.flag==1) :
            self.back_error=np.zeros((mini_batch_size,self.n_in,1))
            for i in xrange(mini_batch_size):
                self.grad_b=self.grad_b+self.delta[i]
                self.grad_w=self.grad_w+np.dot(self.delta[i],
                                               np.transpose(input_2[i].reshape((self.n_in,1))))
                
                self.back_error[i]=np.dot(np.transpose(self.weights),self.delta[i])
            self.back_error=self.back_error.reshape(input_2.shape)
        self.grad=[self.grad_b,self.grad_w]

    def update(self,eta,mini_batch_size):
        self.v_b=0.5*self.v_b -(eta/mini_batch_size)*self.grad_b
        self.biases+=self.v_b
        self.v_w=0.5*self.v_w -(eta/mini_batch_size)*self.grad_w
        self.weights+=self.v_w
        
        



class conv(object):
    def __init__(self,image_shape,filter_shape,activation_func,flag):
        
        self.filter_shape=filter_shape
        self.input_shape=image_shape
        self.act_func=activation_func
        self.flag=flag
        product=self.filter_shape[3]*self.filter_shape[2]*self.filter_shape[1]
        self.weights = np.random.normal(loc=0.0,scale=1.0/np.sqrt(product),size=filter_shape)
        self.biases = np.full((filter_shape[0],1),1.0/np.sqrt(product))
        
                
        self.grad_b=np.zeros(self.biases.shape)
        self.grad_w=np.zeros(self.weights.shape)
        
       
        
        self.v_b=np.zeros(self.biases.shape)
        self.v_w=np.zeros(self.weights.shape)
        
        
    def forward(self, minibatch,mini_batch_size):
        
        size_out=(self.input_shape[2]-self.filter_shape[2]) + 1
        
        self.unpooled_output=np.zeros((mini_batch_size,self.filter_shape[0],size_out,size_out))
        self.unrelued_output=np.zeros((mini_batch_size,self.filter_shape[0],size_out,size_out))
        
        if self.flag==1:
            self.convolution_implmeneted_1(inpt=minibatch,type_="valid",mini_batch_size=mini_batch_size)
        else :
            self.convolution_implmeneted_2(inpt=minibatch,type_="valid",mini_batch_size=mini_batch_size)
        
        
        self.pooling(mini_batch_size)
        self.z=self.output
   

    def convolution_implmeneted_1(self,inpt, type_,mini_batch_size ):
        for j in xrange(mini_batch_size):
            for i in xrange(self.filter_shape[0]):
                self.unrelued_output[j,i]=sg.correlate(inpt[j][0],self.weights[i],type_).reshape(self.unpooled_output.shape[2:])
                + self.biases[i];
                self.unpooled_output[j,i]=self.act_func.fn(self.unrelued_output[j,i])
    def convolution_implmeneted_2(self,inpt, type_,mini_batch_size ):
        for j in xrange(mini_batch_size):
            for i in xrange(self.filter_shape[0]):
                self.unrelued_output[j,i]=sg.correlate(inpt[j],self.weights[i],type_).reshape(self.unpooled_output.shape[2:])
                + self.biases[i];
                self.unpooled_output[j,i]=self.act_func.fn(self.unrelued_output[j,i])
    
    
    def pooling(self,mini_batch_size):
        self.output=np.amax([self.unpooled_output[:,:,(i>>1)&1::2,i&1::2] for i in range(4)],axis=0)
        temp=np.repeat(np.repeat(self.output,2,axis=3),2,axis=2)
        self.map=1.0*(self.unpooled_output==temp)

    def gradient(self,input_1,input_2,mini_batch_size):
        self.delta_1=input_1
        self.grad_b=np.zeros(self.biases.shape)
        self.grad_w=np.zeros(self.weights.shape)
        self.delta_2=np.repeat(np.repeat(self.delta_1,2,axis=3),2,axis=2)
        self.delta_2=(self.map*self.delta_2)*self.act_func.derivative(self.unrelued_output)
        
        n_size=self.delta_2.shape[2]
        
        if (self.flag==2):
            for k in xrange(mini_batch_size):
                for i in xrange(self.delta_2.shape[1]):
                    
                    self.grad_w[i]=self.grad_w[i] + sg.correlate(input_2[k],
                                                                self.delta_2[k,i].reshape(1,n_size,n_size),"valid")
                    self.grad_b[i]=self.delta_2[k,i].sum()
            self.back_error=np.zeros(input_2.shape)
            for k in xrange(mini_batch_size):
                for i in xrange(self.delta_2.shape[1]):
                    self.back_error[k]=self.back_error[k] +  sg.correlate(self.weights[i],
                                                         np.rot90(self.delta_2[k,i],2).reshape((1,n_size,n_size)))
        elif (self.flag==1):
            self.back_error=np.ones((1,0))
            for k in xrange(mini_batch_size):
                for i in xrange(self.delta_2.shape[1]):
                    self.grad_w[i]=self.grad_w[i] + sg.correlate(input_2[k][0],
                                            self.delta_2[k,i].reshape(1,n_size,n_size),"valid")
                    self.grad_b[i]=self.delta_2[k,i].sum()
        #self.grad=[self.grad_b,self.grad_w]
    def update(self,eta,mini_batch_size):
        self.v_b=0.5*self.v_b -(eta/mini_batch_size)*self.grad_b
        self.biases+=self.v_b
        self.v_w=0.5*self.v_w -(eta/mini_batch_size)*self.grad_w
        self.weights+=self.v_w
        
            
class SoftmaxLayer(object):

    def __init__(self, n_in, n_out,activation_func):
        self.n_in = n_in
        self.n_out = n_out
        self.act_func=activation_func
        
        
        self.weights = np.random.normal(loc=0.0, scale=1.0/np.sqrt(n_out+n_in), size=(n_out, n_in))
        self.biases = np.full((n_out,1),1.0/np.sqrt(n_out+n_in))
        
                
        self.grad_b=np.zeros(self.biases.shape)
        self.grad_w=np.zeros(self.weights.shape)
        
       
        self.v_b=np.zeros(self.biases.shape)
        self.v_w=np.zeros(self.weights.shape)
        
        
        self.delta=None
        self.back_error=None
        self.output=None
        
    def forward(self, inpt,mini_batch_size):
        self.output=np.zeros((mini_batch_size,self.n_out,1))
        self.input=inpt
        for i in xrange(mini_batch_size):
            inpt_ = inpt[i].reshape(( self.n_in,1))
            z = np.dot(self.weights,inpt_) + self.biases
            #print z
            z_temp=z-np.max(z)
            #z_temp[z_temp<(-7e+2)]=-7e+2
            softmax= np.exp(z_temp)
            epsilon=0.0000001
            self.output[i]= softmax/(softmax.sum()+epsilon)
    def gradient(self,input_1,input_2,mini_batch_size):
        
        
        self.delta=np.zeros(self.output.shape)
        self.grad_b=np.zeros(self.biases.shape)
        self.grad_w=np.zeros(self.weights.shape)
        
        self.back_error=np.zeros(input_2.shape)
            
        for i in xrange(mini_batch_size):
            
            self.delta[i]=self.output[i]-input_1[i][1]
            self.grad_b=self.grad_b+self.delta[i]
            self.grad_w=self.grad_w+np.dot(self.delta[i],np.transpose(self.act_func.fn(input_2[i])))
            self.back_error[i]=self.act_func.derivative(input_2[i])*(np.dot(np.transpose(self.weights),
                                                                            self.delta[i]))
        self.grad=[self.grad_b,self.grad_w] 
        
    def update(self,eta,mini_batch_size):
        self.v_b=0.5*self.v_b -(eta/mini_batch_size)*self.grad_b
        self.biases+=self.v_b
        self.v_w=0.5*self.v_w -(eta/mini_batch_size)*self.grad_w
        self.weights+=self.v_w
        
        