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
gamma=0.6
punishment=-1
reward_value=1
walking_value=0
episode_size=5

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
        
        self.mlp_s1=nn.Linear(10,500)
        self.mlp_s2=nn.Linear(500,5)
        
        # self.mlp1=nn.Linear(2,50)
        # self.mlp2=nn.Linear(50,1000)
        # self.mlp3=nn.Linear(1000,200)
        # self.mlp4=nn.Linear(200,5) # 5个action
        self.relu=nn.LeakyReLU()
        self.softmax=nn.Softmax()
        self.emb=nn.Embedding(2,10)
    
    def forward(self,x):
        # x1=self.relu(self.mlp2(self.relu(self.mlp1(x))))
        # x2=self.mlp4(self.relu(self.mlp3(x1)))
        # x3=self.softmax(x2)
        # print(f"{out} , {softmax_out}")
        
        x3=self.softmax(self.mlp_s2(self.relu(self.mlp_s1(self.emb(x)))))
        return x3
            
state_list=[]
for i in range(grid_width):
    tmp=[]
    for j in range(grid_height):
        tmp.append(state(i,j))
    state_list.append(tmp)
state_list[2][1].kind=1
state_list[1][0].kind=2
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
            obj.reward[0]=-1
        else:
            obj.reward[0]=init_reward(x,y+1)
        #右
        if(x+1==grid_width):
            obj.reward[1]=-1
        else:
            obj.reward[1]=init_reward(x+1,y)
        #下
        if(y==0):
            obj.reward[2]=-1
        else:
            obj.reward[2]=init_reward(x,y-1)
        #左
        if(x==0):
            obj.reward[3]=-1
        else:
            obj.reward[3]=init_reward(x-1,y)
        #stay
        obj.reward[4]=init_reward(x,y)
    
policy_b=[0.2,0.2,0.2,0.2,0.2]

#generate data

x,y=0,0
x,y=random.randint(0, 4),random.randint(0, 4)
policy_network=Policy_network()
    
alpha_policy=0.0005
main_epoch=3000
# optimizer = optim.Adam(policy_network.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
q_value=policy_network(torch.tensor([0,0],dtype=torch.float32))
policy=torch.argmax(q_value)
print(f"epoch {-1} {q_value} {policy}")
# generate datasest
episode=[]
for i in range(episode_size):
    pv_x,pv_y=x,y
    cur_dir = random.choices(range(len(policy_b)), weights=policy_b)[0]
    reward=state_list[x][y].reward[cur_dir]
    x,y=next_state(x,y,cur_dir)
    episode.append([[pv_x,pv_y],reward,cur_dir,[x,y]])

for main_e in range(main_epoch):
    # print(episode)
    q=0
    for t in range(episode_size):
        reward=episode[-t-1][1]
        q=gamma*q+reward
        # input=torch.zeros(2)
        # input[0]=episode[-t-1][0][0] # x
        # input[1]=episode[-t-1][0][1] # y
        # input.requires_grad_(True)
        input = torch.tensor([[episode[-t-1][0][0], episode[-t-1][0][1]]],dtype=torch.float32)
        action=episode[-t-1][2]
  
        main_output=torch.log(policy_network(input)[0,action])
        main_output.backward()
        # print(main_output)
        if(q>0):
            with torch.no_grad():
                for param in policy_network.parameters():
                    param += alpha_policy * param.grad * q 
                # print(f"{param.grad}  {param.grad.shape}")
        policy_network.zero_grad()
            
    # print(f"Epoch: {main_e}, Loss: {q}")
    # x,y=0,0
    x,y=random.randint(0, 4),random.randint(0, 4)
    # print(f'{x}  {y}')
    episode=[]
    for i in range(episode_size):
        pv_x,pv_y=x,y
        input=torch.zeros(2)
        input[0]=x
        input[1]=y
        policy_b=policy_network(input)
        cur_dir = random.choices(range(5), weights=policy_b)[0]
        reward=state_list[x][y].reward[cur_dir]
        x,y=next_state(x,y,cur_dir)
        episode.append([[pv_x,pv_y],reward,cur_dir,[x,y]])
        
    # print(episode)
    # print("________________________________________")
    q_value=policy_network(torch.tensor([0,0],dtype=torch.float32))
    policy=torch.argmax(q_value)
    print(f"epoch {main_e} {q_value} {policy}")
    
# greed policy
for i in range(grid_width):
    for j in range(grid_height):
        q_value=policy_network(torch.tensor([i,j],dtype=torch.float32))
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