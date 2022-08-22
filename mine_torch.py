from torch.utils import data
import torch
import numpy as np
from torch import nn

def select(X,y,batch_size):
    a=np.random.choice(X.shape[0],batch_size)
    return X[a],y[a]
#随机选择

def calc_acc(X,y,net):
    tot=0
    for i in range(X.shape[0]):
        if(net(X[i]).argmax()==y[i]):
            tot+=1
    return tot/X.shape[0]
#计算准确率

def my_reshape(X,num_input):
    temp=[i[0].reshape(num_input) for i in X]
    res=torch.zeros(len(X),num_input)#形状必须完全一致

    for i in range(len(X)):
        res[i]=temp[i]
    return res
#将MNIST对象转换成tensor

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])
#交叉熵函数实现

def softmax(X):
    X_exp=torch.exp(X)
    partition=X_exp.sum(dim=1,keepdim=True)
    return X_exp/partition#广播机制 自动补全

def init_weight(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,std=0.01,mean=0)
        
def mine_SGD(params,lr,loss_func,epochs,X,y,net):
    for epoch in range(epochs):
        loss=loss_func(net(X),y).mean()
        loss.backward()
    
        with torch.no_grad():#更新时不用计算梯度
            #非常重要
            for param in params:
                param-=(param.grad)*lr
                param.grad.zero_() 
        print(f'epoch {epoch + 1}, loss {loss.item():f}') 