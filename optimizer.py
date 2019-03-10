import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST data/", one_hot=True)
X_train = np.vstack([img.reshape((1,784,1)) for img in mnist.train.images])
y_train = mnist.train.labels
y_train = np.reshape(y_train,(55000,10,1))
X_test = np.vstack([img.reshape(1,784,1) for img in mnist.test.images])
y_test = mnist.test.labels
y_test = np.reshape(y_test,(10000,10,1))
del mnist

class MLP():
    
    def __init__(self,size):
        self.size = size
        self.length = len(size)
    
    def weight_initializer(self):
        biases = [np.random.randn(y, 1) for y in self.size[1:]]
        weights = [np.random.randn(y, x)
                        for x, y in zip(self.size[:-1], self.size[1:])]
        return biases,weights
    
    def relu(self,z):
        return np.maximum(0,z)
    
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def softmax(self,z):
        softz = []
        for i in range(len(z)):
            softz.append(np.exp(z[i])/np.sum(np.exp(z)))
        softz = np.array(softz)
        return softz
    
    def backprop(self,x,y,biases,weights):
        #feedforward
        activation = x
        activations,zs  = [x],[] 
        for b, w in zip(biases,weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        deltalayers,deltaweights,deltabiases=[],[],[]
        deltay = (activations[-1]-y)
        for l in range(1,self.length):
            deltaweight = np.dot(deltay,activations[-l-1].transpose())
            deltaweights.append(deltaweight)
            deltabiase = deltay
            deltabiases.append(deltabiase)
            deltalayer = np.dot(weights[-l].transpose(),deltay)
            deltalayers.append(deltalayer)
            deltay = deltalayer
        #w=self.weights"""
        return activations[-1],deltaweights,deltabiases
    
    def error(self,y,yhat,m,mnb):
        if np.min(yhat)==0:
            print("zero in pic {} batch no {}".format(m,mnb))
        
        #elif np.max(yhat)==1:
        #    print("one in pic {} batch no {}".format(m,mnb))
        return -(y*np.log(yhat))#+(1-y)*np.log(1-yhat))
    
    def traning(self,X_train,y_train,epoch,batch_size,lr,biases,weights,optimizer = 'SGD'):
        l = len(X_train)
        loss=[]
        velbias,velweight,vrw,vrb,sw,sb,rw,rb,t=0,0,0,0,0,0,0,0,0
        for i in range(epoch):
            minibatches = [zip(X_train[k:k+batch_size,:],y_train[k:k+batch_size,:]) for k in range(0,l,batch_size)]
            err=[]
            if optimizer == 'SGD':
                mnb = 0 
                for mini in minibatches:
                    weights,biases,er = self.SGD(mini,weights,biases,lr,mnb)
                    err.append(er)
                    mnb++1
                    
            elif optimizer == 'momentum':
                mnb = 0 
                for mini in minibatches:
                    weights,biases,velweight,velbias,er = self.moment(mini,weights,biases,lr,mnb,velbias,velweight)
                    err.append(er)
                    mnb+=1
                    
            elif optimizer == 'nesterov':
                mnb = 0 
                for mini in minibatches:
                    weights,biases,velweight,velbias,er = self.moment(mini,weights,biases,lr,mnb,velbias,velweight)
                    err.append(er)
                    mnb+=1
                    
            elif optimizer == 'Adagrad':
                mnb = 0 
                for mini in minibatches:
                    weights,biases,er,vrw,vrb = self.Adagrad(mini,weights,biases,lr,vrw,vrb,mnb)
                    err.append(er)
                    mnb+=1
             
            elif optimizer == 'RMSprop':
                mnb = 0 
                for mini in minibatches:
                    weights,biases,er,vrw,vrb = self.RMSprop(mini,weights,biases,lr,vrw,vrb,mnb)
                    err.append(er)
                    mnb+=1
            
            elif optimizer == 'Adam':
                mnb = 0 
                for mini in minibatches:
                    weights,biases,er,sw,sb,rw,rb,t = self.Adam(mini,weights,biases,lr,sw,sb,rw,rb,t,mnb)
                    mnb+=1
                    err.append(er)  
                    
            n = np.sum(err)
            print("epoch {} complete current loss : {}".format(i+1,np.nan_to_num(n)))
            if n>=200:
                pass
            else:
                loss.append(np.nan_to_num(n))
        self.modelweight = weights
        self.modelbiases = biases
        return loss
    
    def testing(self,X_test,y_test):
        self.testlen = len(X_test)
        yout = []
        for x in X_test:
            
            activation = x
            activations = [x] # list to store all the activations, layer by layer
            zs = [] # list to store all the z vectors, layer by layer
            for b, w in zip(self.modelbiases,self.modelweight):
                z = np.dot(w, activation)+b
                zs.append(z)
                activation = self.sigmoid(z)
                activations.append(activation)
            yout.append(activations[-1])
        yhat = np.reshape(yout,(y_test.shape))
        accu = self.accurecy(yhat,y_test)
        print("Test accurecy : {}".format(accu))
        return yhat
    def accurecy(self,yhat,ytest):
        summ=0 
        for i in range(yhat.shape[0]):
            if (np.argmax(yhat[i])) == (np.argmax(ytest[i])):
                summ+=1
        return (summ/self.testlen) 
    
    def SGD(self,batch,weights,biases,lr,mnb):
        err,m=[],0
        for x,y in batch:
            m+=1
            temperror = []
            yhat,deltaweights,deltabiases = self.backprop(x,y,biases,weights)
            #print(deltaweights[1])
            temperror.append(self.error(y,yhat,m,mnb))
            deltaweights,deltabiases = np.array(deltaweights),np.array(deltabiases)
            deltaweights += deltaweights
            deltabiases += deltabiases
        deltaweights = np.array([i for i in list(reversed(deltaweights))])
        deltabiases = np.array([i for i in list(reversed(deltabiases))])
        err.append(np.nan_to_num(np.sum(temperror)/m))
        weights-=lr*(deltaweights/m)
        biases-=lr*(deltabiases/m)
        return weights,biases,err
        
    def moment(self,batch,weights,biases,lr,mnb,velbias=0,velweight=0,moment_para=0.9):
        err,m=[],0
        for x,y in batch:
            m+=1
            temperror = []
            yhat,deltaweights,deltabiases = self.backprop(x,y,biases,weights)
            temperror.append(self.error(y,yhat,m,mnb))
            deltaweights,deltabiases = np.array(deltaweights),np.array(deltabiases)
            deltaweights += deltaweights
            deltabiases += deltabiases
        deltaweights = np.array([i for i in list(reversed(deltaweights))])
        deltabiases = np.array([i for i in list(reversed(deltabiases))])
        err.append(np.nan_to_num(np.sum(temperror)/m))
        weightsnul = (deltaweights/m)
        biasesnul = (deltabiases/m)
        velweight = moment_para*velweight - lr*weightsnul
        velbias = moment_para*velbias - lr*biasesnul 
        weights+= velweight
        biases+= velbias
        return weights,biases,velweight,velbias,err
        
     
    def nesterov(self,batch,weights,biases,lr,mnb,velbias=0,velweight=0,moment_para=0.9):
        err,m=[],0
        biases,weights = biases + moment_para*velbias ,weights + moment_para*velweight
        for x,y in batch:
            m+=1
            temperror = []
            yhat,deltaweights,deltabiases = self.backprop(x,y,biases,weights)
            temperror.append(self.error(y,yhat,m,mnb))
            deltaweights,deltabiases = np.array(deltaweights),np.array(deltabiases)
            deltaweights += deltaweights
            deltabiases += deltabiases
        deltaweights = np.array([i for i in list(reversed(deltaweights))])
        deltabiases = np.array([i for i in list(reversed(deltabiases))])
        err.append(np.nan_to_num(np.sum(temperror)/m))
        weightsnul = (deltaweights/m)
        biasesnul = (deltabiases/m)
        velweight = moment_para*velweight - lr*weightsnul
        velbias = moment_para*velbias - lr*biasesnul 
        weights+= velweight
        biases+= velbias
        return weights,biases,velweight,velbias,err
    
    def Adagrad(self,batch,weights,biases,lr,vrw,vrb,mnb,delta=0.0000001):
        err,m=[],0
        for x,y in batch:
            m+=1
            temperror = []
            yhat,deltaweights,deltabiases = self.backprop(x,y,biases,weights)
            temperror.append(self.error(y,yhat,m,mnb))
            deltaweights,deltabiases = np.array(deltaweights),np.array(deltabiases)
            deltaweights += deltaweights
            deltabiases += deltabiases
        deltaweights = np.array([i for i in list(reversed(deltaweights))])
        deltabiases = np.array([i for i in list(reversed(deltabiases))])
        err.append(np.nan_to_num(np.sum(temperror)/m))
        weightsnul = (deltaweights/m)
        biasesnul = (deltabiases/m)
        vrw += weightsnul*weightsnul
        vrb += biasesnul*biasesnul
        for i in range(len(weightsnul)):
            
            weights[i] -= (lr/(delta+np.sqrt(vrw[i])))*weightsnul[i] 
            biases[i] -= (lr/(delta+np.sqrt(vrb[i])))*biasesnul[i] 
        return weights,biases,err,vrw,vrb
    
    def RMSprop(self,batch,weights,biases,lr,vrw,vrb,mnb,decay_rate=0.7,delta=0.000001):
        err,m=[],0
        for x,y in batch:
            m+=1
            temperror = []
            yhat,deltaweights,deltabiases = self.backprop(x,y,biases,weights)
            temperror.append(self.error(y,yhat,m,mnb))
            deltaweights,deltabiases = np.array(deltaweights),np.array(deltabiases)
            deltaweights += deltaweights
            deltabiases += deltabiases
        deltaweights = np.array([i for i in list(reversed(deltaweights))])
        deltabiases = np.array([i for i in list(reversed(deltabiases))])
        err.append(np.nan_to_num(np.sum(temperror)/m))
        weightsnul = (deltaweights/m)
        biasesnul = (deltabiases/m)
        vrw = decay_rate*vrw + (1-decay_rate)*weightsnul*weightsnul
        vrb = decay_rate*vrb + (1-decay_rate)*biasesnul*biasesnul
        for i in range(len(weightsnul)):
            
            weights[i] -= (lr/(delta+np.sqrt(vrw[i])))*weightsnul[i] 
            biases[i] -= (lr/(delta+np.sqrt(vrb[i])))*biasesnul[i] 
        return weights,biases,err,vrw,vrb
    
    def Adam(self,batch,weights,biases,lr,sw,sb,rw,rb,t,mnb,p1=0.9,p2=0.999,delta=0.000001):
        err,m,yout=[],0,[]
        for x,y in batch:
            m+=1
            temperror = []
            yhat,deltaweights,deltabiases = self.backprop(x,y,biases,weights)
            temperror.append(self.error(y,yhat,m,mnb))
            yout.append(yhat)
            deltaweights,deltabiases = np.array(deltaweights),np.array(deltabiases)
            deltaweights += deltaweights
            deltabiases += deltabiases
        deltaweights = np.array([i for i in list(reversed(deltaweights))])
        deltabiases = np.array([i for i in list(reversed(deltabiases))])
        err.append(np.nan_to_num(np.sum(temperror)/m))
        weightsnul = (deltaweights/m)
        biasesnul = (deltabiases/m)
        t+=1
        sw = p1*sw + (1-p1)*weightsnul
        sb = p1*sb + (1-p1)*biasesnul
        rw = p2*rw + (1-p2)*weightsnul*weightsnul
        rb = p2*rb + (1-p2)*biasesnul*biasesnul
        swhat = sw/(1-np.power(p1,t))
        rwhat = rw/(1-np.power(p2,t))
        sbhat = sb/(1-np.power(p1,t))
        rbhat = rb/(1-np.power(p2,t))
        for i in range(len(weightsnul)):
            weights[i] -= lr*(swhat[i]/(np.sqrt(rwhat[i])+delta))
            biases[i] -= lr*(sbhat[i]/(np.sqrt(rbhat[i])+delta))
        return weights,biases,err,sw,sb,rw,rb,t
    
model = MLP([784,512,10])
biases,weights = model.weight_initializer()
print("SGD ====>")
sgdloss= model.traning(X_train[0:1000,:],y_train[0:1000,:],10,100,0.01,biases,weights,optimizer = 'SGD')
model.testing(X_test[0:50],y_test[0:50])

biases,weights = model.weight_initializer()
print("Momentum ====>")
momenloss= model.traning(X_train[0:1000,:],y_train[0:1000,:],10,100,0.01,biases,weights,optimizer = 'momentum')
model.testing(X_test[0:50],y_test[0:50])

biases,weights = model.weight_initializer()
print("Nesterov ====>")
nestloss= model.traning(X_train[0:1000,:],y_train[0:1000,:],10,100,0.01,biases,weights,optimizer = 'nesterov')
model.testing(X_test[0:50],y_test[0:50])

biases,weights = model.weight_initializer()
print("Adagrad ====>")
adagloss= model.traning(X_train[0:1000,:],y_train[0:1000,:],10,100,0.01,biases,weights,optimizer = 'Adagrad')
model.testing(X_test[0:50],y_test[0:50])

biases,weights = model.weight_initializer()
print("RMSprop ====>")
rmsloss= model.traning(X_train[0:1000,:],y_train[0:1000,:],10,100,0.01,biases,weights,optimizer = 'RMSprop')
model.testing(X_test[0:50],y_test[0:50])

biases,weights = model.weight_initializer()
print("Adam ====>")
adamloss= model.traning(X_train[0:1000,:],y_train[0:1000,:],10,100,0.01,biases,weights,optimizer = 'Adam')
model.testing(X_test[0:50],y_test[0:50])

plt.plot(sgdloss)
plt.plot(momenloss)
plt.plot(nestloss)
plt.plot(adagloss)
plt.plot(rmsloss)
plt.plot(adamloss)
plt.ylabel("error")
plt.xlabel("epoch")
plt.legend(['SGD','Mmentum','Nasterov','Adagrad','RMSprop','Adam'])
plt.show()






