


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