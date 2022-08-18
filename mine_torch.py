from torch.utils import data
import torchvision
import numpy as np

def select(X,y,batch_size):
    a=np.random.choice(X.shape[0],batch_size)
    return X[a],y[a]
#随机选择

