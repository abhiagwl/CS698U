
import random
import numpy as np
import math

#activation functions

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

class sigmoid(object):
    @staticmethod
    def fn(z):
        return 1/(1+np.exp(-z))
    @staticmethod
    def derivative(z):
        return (1/(1+np.exp(-z)))*(1-(1/(1+np.exp(-z))))

#gradient versions


class vanilla(object):
    @staticmethod
    def cupdate(w,dw,vw,cw,eta,n):
        return cw
    @staticmethod
    def vupdate(w,dw,vw,cw,eta,n):
        return vw
    @staticmethod
    def wupdate(w,dw,vw,cw,eta,n):
        return w-(eta/n)*dw

class momentum(object):
    @staticmethod
    def cupdate(w,dw,vw,cw,eta,n):
        return cw
    @staticmethod
    def vupdate(w,dw,vw,cw,eta,n):
        return vw*0.5-(eta/n)*dw
    @staticmethod
    def wupdate(w,dw,vw,cw,eta,n):
        return w+vw

class adgrad(object):
    @staticmethod
    def cupdate(w,dw,vw,cw,eta,n):
        return cw+dw**2
    @staticmethod
    def vupdate(w,dw,vw,cw,eta,n):
        return vw
    @staticmethod
    def wupdate(w,dw,vw,cw,eta,n):
        return w-((eta/n)*dw)/(np.sqrt(cw)+math.pow(10,-6))

class rmsprop(object):
    @staticmethod
    def cupdate(w,dw,vw,cw,eta,n):
        return 0.9*cw+(1-0.9)*dw**2
    @staticmethod
    def vupdate(w,dw,vw,cw,eta,n):
        return vw
    @staticmethod
    def wupdate(w,dw,vw,cw,eta,n):
        return w-((eta/n)*dw)/(np.sqrt(cw)+math.pow(10,-6))




"""
Network is the main class to execute the different methods
Code currently implements the neural network architecture with different activation functions and choice of different gradient options. 
The last layer is softmax and the loss function is cross entropy as these are pretty standard choices.
The use of regularization has been omitted as the purpose was to study the different techniques and compare how they fare against each other
"""


class Network(object):

    def __init__(self, n_layers, nnodes, actfun=sigmoid,gradient= vanilla):
        """
        Whenever an instance of the class is called the following parameters are required:

        n_layers : total number of layers (Example:   4)
        nnodes   : a list having the number of nodes in the different layers. (Example:   [784,30,30,10]) 
                   here the first layer is the input and the last one is output. The remaining layers are the hidden layers
        actfun   : takes one of the three possible activation functions namely tanh, ReLU and sigmoid
        gradient : takes one of the few possible gradient approaches, namely rmsprop,momentum,adagrad and vanilla gd
        
        weights and biases: initializes the weights and biases of the neurons depending upon the network architecture
                            The divide by sqrt(x) is for an efficient initialization of weights()
        
        velocity_biases and velocity_weights : initialize array with shape same as that of the weights and biases 
                                               to facilitate the 
                                               momentum and related gradients
        
        cache_biases and cache_weights : same as the velocity_biases & wieghts for the adagrad and rmsprop gradients


        This redundant update rule is done to make the whole process general and easier to run

        """


        self.num_layers =n_layers 
        self.sizes = nnodes
        
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.velocity_biases = [np.zeros((y, 1)) for y in self.sizes[1:]]
        
        self.velocity_weights = [np.zeros((y, x))
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.cache_biases = [np.zeros((y, 1)) for y in self.sizes[1:]]
        
        self.cache_weights = [np.zeros((y, x))
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.actfun=actfun
        self.gradient=gradient

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            z=np.dot(w, a)+b
            a = self.actfun.fn(z)
        sftmx=np.exp(z-np.max(z))
        a= sftmx/np.sum(sftmx) #changes the last activation unit to softmax version
        return a

    def train(self, training_data, epochs, mini_batch_size, eta,evaluation_data):
        """
        The mini-batch approach is used to update the weights in the network. 
        The train function executes the training of the network for the given number of
        epochs as the argument.
        """
        n = len(training_data)
        evaluation_acc=[]
        train_acc=[]
        evaluation_cost=[]
        train_cost=[]
        for j in xrange(epochs):

            random.shuffle(training_data)

            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta,len(training_data),training_data)                

            print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(evaluation_data), len(evaluation_data))
            evaluation_acc.append((float(self.evaluate(evaluation_data))/len(evaluation_data))*100)
            train_acc.append((float(self.evaluate(training_data))/len(training_data))*100)
            evaluation_cost.append(self.cost(evaluation_data))
            train_cost.append(self.cost(training_data))
        return [evaluation_acc,evaluation_cost,train_acc,train_cost]
           

    def update_mini_batch(self, mini_batch, eta,n,training_data ):

        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_gradient_b, delta_gradient_w = self.backprop(x, y,training_data)
            gradient_b = [nb+dnb for nb, dnb in zip(gradient_b, delta_gradient_b)]
            gradient_w = [nw+dnw for nw, dnw in zip(gradient_w, delta_gradient_w)]
        
        self.cache_weights = [self.gradient.cupdate(w,nw,vw,cw,eta,len(mini_batch))
                        for w, nw, vw,cw in zip(self.weights, gradient_w,self.velocity_weights,self.cache_weights)]
        self.cache_biases = [self.gradient.cupdate(b,nb,vb,cb,eta,len(mini_batch))
                       for b, nb,vb,cb in zip(self.biases, gradient_b,self.velocity_biases,self.cache_biases)]
        self.velocity_weights = [self.gradient.vupdate(w,nw,vw,cw,eta,len(mini_batch))
                        for w, nw, vw,cw in zip(self.weights, gradient_w,self.velocity_weights,self.cache_weights)]
        self.velocity_biases = [self.gradient.vupdate(b,nb,vb,cb,eta,len(mini_batch))
                       for b, nb,vb,cb in zip(self.biases, gradient_b,self.velocity_biases,self.cache_biases)]
        self.weights = [self.gradient.wupdate(w,nw,vw,cw,eta,len(mini_batch))
                        for w, nw, vw,cw in zip(self.weights, gradient_w,self.velocity_weights,self.cache_weights)]
        self.biases = [self.gradient.wupdate(b,nb,vb,cb,eta,len(mini_batch))
                       for b, nb,vb,cb in zip(self.biases, gradient_b,self.velocity_biases,self.cache_biases)]


                       
    

    def backprop(self, x, y,training_data):

        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        
        # feedforward
        activation = x
        layer_outputs_ = [x] # stores all the layer outputs
        layer_zs_ = []       # stores all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            layer_zs_.append(z)
            activation = self.actfun.fn(z)
            layer_outputs_.append(activation)
        
        #to implement softmax layer at the final output node
        layer_outputs_.pop()
        softmx = np.exp(layer_zs_[-1]-np.max(layer_zs_[-1]))  #np.max(layer_zs_[-1]) is done to keep the values in check. 
                                                              #It does not changes the values of the final answer.
        layer_outputs_.append(softmx / softmx.sum())
        # backward pass
        # The error or delta calculation for the cross entropy loss and final layer being softmax reduces to a neat expression 
        # that is the difference in the output/predicted and the real y values

        delta = (layer_outputs_[-1]-y) 
        # Now the erorr from the last layer is linked to the gradients of weights and the biases

        gradient_b[-1] = delta
        gradient_w[-1] = np.dot(delta, layer_outputs_[-2].transpose())
        
        # Now this algorithm propagates the error for all the layer except the last one as it is input and hence no error there

        for l in xrange(2, self.num_layers):
            z = layer_zs_[-l]
            activation_derivative = self.actfun.derivative(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * activation_derivative
            gradient_b[-l] = delta
            gradient_w[-l] = np.dot(delta, layer_outputs_[-l-1].transpose())
        return (gradient_b, gradient_w)


    def evaluate(self, data):

        results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)


    def cost(self, data ):
        cost = 0.0
        for x,y in data:
            a=self.feedforward(x)
            cost+= (np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a))))/len(data)
        return cost

    def cost_indvdl(self, x,y):
        cost = 0.0
        a=self.feedforward(x)
        cost= np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
        return cost

    def grad_check(self, training_data):

        random.shuffle(training_data)
        data=training_data[0]
        backprop_db = [np.zeros(b.shape) for b in self.biases]
        backprop_dw = [np.zeros(w.shape) for w in self.weights]
        backprop_db,backprop_dw = self.backprop(data[0],data[1],training_data)
        h=math.pow(10,-6)
        estimate=np.zeros(self.biases[0].shape)
        for i,value in np.ndenumerate(self.biases[0]):
            epsilon = np.zeros(self.biases[0].shape)
            epsilon[i]=h
            self.biases[0]+=epsilon
            cost1=self.cost_indvdl(data[0],data[1] )
            self.biases[0]=self.biases[0]-2*epsilon
            cost2=self.cost_indvdl(data[0],data[1] )
            estimate[i]=(cost1-cost2)/(h)
            self.biases[0]+=epsilon
        #print backprop_db[1]
        return estimate
    
    """
    Sources : Internet in general
              Micheal Neilsen's neuralnetworksanddeeplearning.com 
              CS231 : Stanford Lecture Notes
              StackExchange
    """