from torch.utils import data
import torchvision

def load_array(data_array,batch_size,is_shuffle=True,resize=None):
    return data.DataLoader(data_array,batch_size,shuffle=is_shuffle)
    #是否随机挑选batch_size个