import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from tqdm import trange
import pickle

global shape_tracker
shape_tracker=[]

def add2trackerdict(val):
    shape_tracker.append(val)


def load_dataset(flatten=False):
    (X_train,y_train),(X_test,y_test)=mnist.load_data()

    X_train=X_train.astype(float)/255
    X_test=X_test.astype(float)/255

    X_train,X_val=X_train[:-1000],X_train[-1000:]
    y_train,y_val=y_train[:-1000],y_train[-1000:]

    if flatten:
        X_train=X_train.reshape([X_train.shape[0],-1])
        X_val=X_val.reshape([X_val.shape[0],-1])
        X_test=X_test.reshape([X_test.shape[0],-1])
    return X_train,y_train,X_val,y_val,X_test,y_test



def softmax_crossentropy_with_logits(logits,reference_answer):
    logits_for_answer=logits[np.arange(len(logits)),reference_answer]
    xentropy=-logits_for_answer+np.log(np.sum(np.exp(logits),axis=-1))
    return xentropy

def grad_softmax_crossentropy_with_logits(logits,reference_answer):
    ones_for_answers=np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)),reference_answer]=1
    softmax=np.exp(logits)/np.exp(logits).sum(axis=-1,keepdims=True)
    return (-ones_for_answers+softmax)/logits.shape[0]

class Dense:
    def __init__(self,input_dim=False,n_units=0,learning_rate=0.1):

        if input_dim:
            self.learning_rate=learning_rate
            self.weights=np.random.randn(input_dim,n_units)*0.01
            self.bias=np.zeros(n_units)
            add2trackerdict(n_units)
        else:
            self.learning_rate=learning_rate
            self.weights=np.random.randn(shape_tracker[-1],n_units)*0.01
            self.bias=np.zeros(n_units)
            add2trackerdict(n_units)

    def feed_forward(self,input):
        return np.matmul(input,self.weights)+self.bias

    def backprop(self,x,grad_later_layer):
        grad2bpassed=np.dot(grad_later_layer,self.weights.T)
        deltaWd=np.dot(grad_later_layer.T,x).T
        deltaWdb=np.sum(grad_later_layer)
        self.weights=self.weights-self.learning_rate*deltaWd
        self.bias=self.bias-self.learning_rate*deltaWdb
        return grad2bpassed

class ReLU:
    def __init__(self):
        pass
    def feed_forward(self,x):
        return np.maximum(0,x)
    def backprop(self,x,grad_out):
        return grad_out*(x>0)


class Sequential:
    def __init__(self):
         self.model=[]
         self.train_log=[]
         self.val_log=[]


    def add(self,obj):
        if len(shape_tracker)==0:
            raise Exception("Input shape not specified.")
        else:
            self.model.append(obj)

    def feed_forward(self,X):
        fpassstack=[]
        input=X
        for i in range(len(self.model)):
            fpassstack.append(self.model[i].feed_forward(X))
            X=self.model[i].feed_forward(X)
        return fpassstack

    def predict(self,X):
        logits=self.feed_forward(X)[-1]
        return logits.argmax(axis=-1)

    def train(self,X,y):
        layer_activations=self.feed_forward(X)
        logits=layer_activations[-1]

        loss=softmax_crossentropy_with_logits(logits,y)
        loss_grad=grad_softmax_crossentropy_with_logits(logits,y)

        for i in range(1,len(self.model)):
            loss_grad=self.model[len(self.model)-i].backprop(layer_activations[len(self.model)-i-1],loss_grad)
        return np.mean(loss)

    def interate_minibatches(self,inputs,targets,batch_size,shuffle=False):
        assert len(inputs)==len(targets)
        if shuffle:
            indices=np.random.permutation(len(inputs))
        for start_idx in trange(0,len(inputs)-batch_size+1,batch_size):
            if shuffle:
                excerpt=indices[start_idx:start_idx+batch_size]
            else:
                excerpt=slice(start_idx,start_idx+batch_size)
            yield inputs[excerpt],targets[excerpt]

    def fit(self,X_train,y_train,epochs,batch_size):
        for epoch in range(epochs):
            for x_t,y_t in self.interate_minibatches(X_train,y_train,batch_size=batch_size,shuffle=True):
                self.train(x_t,y_t)
            self.train_log.append(np.mean(self.predict(X_train)==y_train))
            self.val_log.append(np.mean(self.predict(X_val)==y_val))

            print("Epoch:%d Train accuaracy=%f Val accuaracy=%f"%(epoch,self.train_log[-1],self.val_log[-1]))

    def save(self,name):
        pickle.dump(self,open(f"{name}","wb"))

def load_model(dirn):
    with open(f"{dirn}",'rb') as fp:
        model=pickle.load(fp)
    return model


X_train,y_train,X_val,y_val,X_test,y_test=load_dataset(flatten=True)

model=Sequential()
model.add(Dense(input_dim=X_train.shape[1],n_units=100))
model.add(ReLU())
model.add(Dense(n_units=200))
model.add(ReLU())
model.add(Dense(n_units=10))

model.fit(X_train,y_train,epochs=25,batch_size=32)
model.save('Model1.pkl')

model2=load_model('Model1.pkl')
print(model2.predict(X_test[0:10]))
