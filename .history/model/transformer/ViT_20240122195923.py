import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import torchvision
device=torch.device('cuda')

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super(MultiHeadAttention, self).__init__() #初始化 nn.Module 
        assert d_model % num_heads == 0 # 能够等分 h 为头数目
        self.num_heads=num_heads
        self.d_model=d_model
        self.d_k = d_model // num_heads # key 通过类似CNN的多通道机制进行分离
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k) # 表示对倒数第二个和最后一个维度进行转置。
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9) # 传入mask
        attn_probs = torch.softmax(attn_scores, dim=-1) #对dk进行
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)#先分成两个维度 不破坏原始数据结构
        # (batch_size, self.h, seq_length, self.d_k)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, _ = x.size() #split 的逆向操作
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model) 
        #.contiguous() 可以确保张量在内存中是按照顺序排列的，以便后续的操作。
    
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model) # position ecoding 矩阵，对小于最大长度所有序列计算
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1) #插入维度 batch_size
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0)) #它会与模型的参数一起被 PyTorch 的 state_dict() 保存和加载
         # 同时加了一维 batch_size self.pe 是一个形状为 (1, max_seq_length, d_model) 的张量
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)] # 利用广播机制相加 x.size(1) 说明超出seq_length 的部分不加
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model) #最后一个维度
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output)) # 残差连接
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

    

device = torch.device('cuda')

def process_data(X,device=torch.device('cuda')):
    temp_x=[i[0][0].unsqueeze_(0) for i in X]#升维再导入
    data_x=torch.cat(temp_x).reshape(-1,28,28)#加上通道数
    data_y=torch.tensor([i[1] for i in X])
    return data_x.to(device),data_y.to(device)

trans = torchvision.transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../../data", train=False, transform=trans, download=True)

(train_x0,train_y0)=process_data(mnist_train,device)
(test_x0,test_y0)=process_data(mnist_test,device)
patch_size=4
train_x=train_x0.unsqueeze(1)
train_x=nn.functional.unfold(train_x,kernel_size=(patch_size,patch_size),stride=patch_size).transpose(-1,-2).to(device)

transformer=EncoderLayer(16,2,16,0.1)
print(transformer(train_x))