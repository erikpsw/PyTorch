import taichi as ti
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init

ti.init(ti.cpu)
pixel_size=60
grid_width=5
grid_height=5
width=pixel_size*grid_width
height=pixel_size*grid_height
step=1/grid_width
gamma=0.9
punishment=-1
reward_value=3
walking_value=0
episode_size=4
wall_value=0
# sample_size=10

#网格图类
class grid:
    def __init__(self,width,height) -> None:
        self.canvas=np.ones((width,height,3))

    def set_color(self,x,y,color):
        self.canvas[x*pixel_size:(x+1)*pixel_size,y*pixel_size:(y+1)*pixel_size,:]=color

#智能体
class agent:
    def __init__(self,x,y) -> None:
        self.pos=[(x+0.5)/grid_width,(y+0.5)/grid_height]

    def move(self,dir):
        if(dir==0):#上
            self.pos[1]+=step
        elif(dir==1):#右
            self.pos[0]+=step  
        elif(dir==2):#下
            self.pos[1]-=step
        elif(dir==3):#左
            self.pos[0]-=step
        elif(dir==4):
            pass
class state:
    def __init__(self,x,y) -> None:
        self.x=x
        self.y=y
        self.reward=[0,0,0,0,0]
        self.kind=0#1为目标，2为障碍
        self.policy=0
        self.value=0
        self.q_value=[0,0,0,0,0]
        
    def __repr__(self):
        return f"[{self.x},{self.y},reward{self.reward}]"
    
def init_reward(x,y):
    state=state_list[x][y].kind
    if(state==1):
        return reward_value
    elif(state==2):
        return punishment
    else:
        return walking_value
    
def next_state(x,y,dir):
    if(dir==4):
        return x,y
    elif(dir==0):
        if(y+1==grid_height):
            return x,y
        else:
            return x,y+1
    #右
    elif(dir==1):
        return (x,y) if x+1==grid_width else (x+1,y)
    
    #下
    elif(dir==2):
        return (x,y) if y==0 else (x,y-1)

    #左
    elif(dir==3):
        return (x,y) if x==0 else (x-1,y)
    
class Policy_network(nn.Module):
    def __init__(self):
        super(Policy_network, self).__init__() #初始化 nn.Module 
        
        # self.mlp_s1=nn.Linear(2,300)
        # self.mlp_s2=nn.Linear(300,5)
        
        # self.mlp1=nn.Linear(10,100)
        # self.mlp2=nn.Linear(100,400)
        # self.mlp3=nn.Linear(400,10)
        # self.mlp4=nn.Linear(10,5) # 5个action
        
        self.mlp1=nn.Linear(10,1000)
        self.mlp2=nn.Linear(1000,5)
        # self.drop=nn.Dropout(0.1)
        self.relu=nn.LeakyReLU()
        self.sigmoid=nn.Sigmoid()
        self.atan=nn.Tanh()
        
    def forward(self,x):
        # out=self.relu(self.mlp2(self.relu(self.mlp1(x))))
        # out=self.mlp4(self.atan(self.mlp3(out)))
        
        out=self.mlp2(self.relu(self.mlp1(x)))
        
        softmax_out=F.softmax(out,dim=0)
        # print(f"{out} , {softmax_out}")
        return softmax_out
        
class Value_network(nn.Module):
    def __init__(self):
        super(Value_network, self).__init__() #初始化 nn.Module 
        
        # self.mlp_s1=nn.Linear(7,300)
        # self.mlp_s2=nn.Linear(300,1)
        self.relu = nn.LeakyReLU()
        
        # self.mlp1=nn.Linear(10,50)
        # self.mlp2=nn.Linear(50,200)
        # self.mlp3=nn.Linear(200,10)
        # self.mlp4=nn.Linear(10,1) # 5个action
        
        self.mlp1=nn.Linear(10,600)
        self.mlp2=nn.Linear(600,1)

    
    def forward(self,x):
        # x=self.relu(self.mlp2(self.relu(self.mlp1(x))))
        # x=self.mlp4(self.relu(self.mlp3(x)))
        
        # x = self.relu(self.mlp_s1(x))  # 在第一个线性层后应用ReLU
        # x = self.mlp_s2(x)
        
        x=self.mlp2(self.relu(self.mlp1(x)))
        
        return x
            
state_list=[]
for i in range(grid_width):
    tmp=[]
    for j in range(grid_height):
        tmp.append(state(i,j))
    state_list.append(tmp)
state_list[2][1].kind=1
state_list[1][0].kind=0
state_list[1][1].kind=0
state_list[1][3].kind=0
state_list[1][2].kind=0
state_list[2][3].kind=0
state_list[3][1].kind=0
state_list[4][4].kind=0

for i in range(grid_width):
    tmp=[]
    for j in range(grid_height):
        obj=state_list[i][j]
        x=obj.x
        y=obj.y
        #上
        if(y+1==grid_height):
            obj.reward[0]=wall_value
        else:
            obj.reward[0]=init_reward(x,y+1)
        #右
        if(x+1==grid_width):
            obj.reward[1]=wall_value
        else:
            obj.reward[1]=init_reward(x+1,y)
        #下
        if(y==0):
            obj.reward[2]=wall_value
        else:
            obj.reward[2]=init_reward(x,y-1)
        #左
        if(x==0):
            obj.reward[3]=wall_value
        else:
            obj.reward[3]=init_reward(x-1,y)
        #stay
        obj.reward[4]=init_reward(x,y)
    
policy_b=[0.2,0.2,0.2,0.2,0.2]

#generate data

policy_network=Policy_network()
value_network=Value_network()
# for param in policy_network.parameters():
#     init.constant_(param, 0.001)
# for param in value_network.parameters():
#     init.constant_(param, 0.001)
policy_alpha=0.001
value_alpha=0.001
main_epoch=2000
# optimizer = optim.Adam(policy_network.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
start=torch.zeros(10)
start[0]=1.
start[5]=1.

for main_e in range(main_epoch):
    # on-policy 算法
    # episode=[]
    # x,y=random.randint(0, 4),random.randint(0, 4)
    x,y=0,0
    for i in range(episode_size):
        pv_x,pv_y=x,y
        input=torch.zeros(10)
        input[x]=1.
        input[5+y]=1.
        policy_b=policy_network(input)
        # print(policy_b)
        cur_dir = random.choices(range(len(policy_b)), weights=policy_b)[0]
        reward=state_list[x][y].reward[cur_dir]
        x,y=next_state(x,y,cur_dir)
        
        # episode.append([[pv_x,pv_y],cur_dir,reward,[x,y]])
    # print(episode)
    # for i in range(sample_size):
    #     data=random.sample(episode,1)[0]
    #     # print(data)
    #     st=torch.tensor(data[0],dtype=torch.float32)
    #     stp=torch.tensor(data[3],dtype=torch.float32)
    #     cur_v=value_network(st)[0]
    #     delta_t=data[2]+gamma*value_network(stp)[0]-cur_v
        # print(delta_t)
        # cur_v.backward()
        # with torch.no_grad():
        #     for param in value_network.parameters():
        #         param += value_alpha * param.grad * delta_t
        #         param.grad.zero_()
    # for i in range(sample_size):
    #     data=random.sample(episode,1)[0]
        # print(data)
        # st=torch.tensor(data[0],dtype=torch.float32)
        # stp=torch.tensor(data[3],dtype=torch.float32)
        # cur_v=value_network(st)
        # delta_t=data[2]+gamma*value_network(stp)-cur_v
        # print(delta_t)
        # print(data[1])
        # main_output=torch.log(policy_network(st)[data[1]])
        # st=torch.tensor([pv_x,pv_y],dtype=torch.float32)
        # stp=torch.tensor([x,y],dtype=torch.float32)
        
        st=torch.zeros(10)
        stp=torch.zeros(10)
        st[pv_x]=1.
        st[5+pv_y]=1.
        stp[x]=1.
        stp[5+y]=1.
        cur_v=value_network(st)
        delta_t=reward+gamma*value_network(stp)-cur_v
        main_output=torch.log(policy_network(stp)[cur_dir])
        main_output.backward()
        cur_v.backward()
        # if delta_t>0:
        with torch.no_grad():
            for param in policy_network.parameters():
                param += policy_alpha * param.grad * delta_t
                # print(delta_t)
                param.grad.zero_()
            for param in value_network.parameters():
                param += value_alpha * param.grad * delta_t
                # print(param.grad)
        policy_network.zero_grad()
        value_network.zero_grad()
            
        
    print(f"Epoch: {main_e}, Policy: {policy_network(start).tolist()}")


# greed policy
for i in range(grid_width):
    for j in range(grid_height):
        input=torch.zeros(10)
        input[i]=1.
        input[5+j]=1.
        q_value=policy_network(input)
        policy=torch.argmax(q_value)
        print(f"{i} {j} {q_value} {policy}")
        state_list[i][j].policy=policy
        
line_b=np.array([[x,0] for x in range(pixel_size,width,pixel_size)])/width
line_b1=np.array([[0,y] for y in range(pixel_size,height,pixel_size)])/height
line_e=np.array([[x,width-1] for x in range(pixel_size,width,pixel_size)])/width
line_e1=np.array([[height-1,y] for y in range(pixel_size,height,pixel_size)])/height
grid_world=grid(width,height)
for i in range(grid_width):
    for j in range(grid_height):
        if(state_list[i][j].kind==1):
            grid_world.set_color(i,j,[120/255,152/255,232/255])
        elif(state_list[i][j].kind==2):
            grid_world.set_color(i,j,[245/255,151/255,148/255])

agent_x=0
agent_y=0
myagent=agent(agent_x,agent_y)

dt=0.5
N=0

gui=ti.GUI("grid",(width,height))

while gui.running:
    N+=1
    if(N==dt*60):
        N=0
        dir=state_list[agent_x][agent_y].policy
        agent_x,agent_y=next_state(agent_x,agent_y,dir)
        myagent=agent(agent_x,agent_y)
    gui.set_image(grid_world.canvas)
    gui.lines(begin=np.concatenate((line_b,line_b1),axis=0), end=np.concatenate((line_e,line_e1),axis=0), radius=1, color=0x000000)
    gui.circle(myagent.pos,color=0x000000,radius=5)
    for i in range(grid_width):
        for j in range(grid_height):
            gui.text(content=str(state_list[i][j].value)[:4], pos=[i/grid_width,(j+1)/grid_height], font_size=20, color=0x000000)
    gui.show()