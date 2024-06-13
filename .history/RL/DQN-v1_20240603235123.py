import taichi as ti
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

ti.init(ti.cpu)
pixel_size=60
grid_width=5
grid_height=5
width=pixel_size*grid_width
height=pixel_size*grid_height
step=1/grid_width
gamma=0.9
punishment=-5
dataset_size=1000

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
        return 1
    elif(state==2):
        return punishment
    else:
        return 0
    
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
    
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__() #初始化 nn.Module 
        
        # self.mlp_s1=nn.Linear(2,300)
        # self.mlp_s2=nn.Linear(300,5)
        
        self.mlp1=nn.Linear(2,40)
        self.mlp2=nn.Linear(40,300)
        self.mlp3=nn.Linear(300,50)
        self.mlp4=nn.Linear(50,5) # 5个action
        self.relu=nn.ReLU()
    
    def forward(self,x):
        return self.mlp4(self.relu(self.mlp3(self.relu(self.mlp2(self.relu(self.mlp1(x)))))))
        # return self.mlp_s2(self.mlp_s1(x))
            
state_list=[]
for i in range(grid_width):
    tmp=[]
    for j in range(grid_height):
        tmp.append(state(i,j))
    state_list.append(tmp)
state_list[2][1].kind=1
state_list[1][0].kind=2
state_list[1][1].kind=2
state_list[1][3].kind=2
state_list[1][2].kind=2
state_list[2][3].kind=2
state_list[3][1].kind=2
state_list[4][4].kind=2

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
target_network=Network()
main_network=Network()
learning_epoch=50
main_epoch=1
epsilon=0.6
epochs=10
mse = nn.MSELoss()
optimizer = optim.Adam(main_network.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

# generate datasest
dataset=[]
for i in range(dataset_size):
    pv_x,pv_y=x,y
    cur_dir = random.choices(range(len(policy_b)), weights=policy_b)[0]
    r=state_list[pv_x][pv_y].reward[cur_dir]
    x,y=next_state(x,y,cur_dir)
    dataset.append([[pv_x,pv_y],cur_dir,r,[x,y]])

batch_size=100
for main_e in range(main_epoch):
    for epoch in range(epochs):
        
        batch=random.sample(dataset,batch_size)
        input=torch.zeros((batch_size,2))
        next_input=torch.zeros((batch_size,2)) # 下一个状态输入
        R=torch.zeros(batch_size)
        A=torch.zeros(batch_size)
        for i,data in enumerate(batch):
            input[i,0]=data[0][0] # x
            input[i,1]=data[0][1] # y
            next_input[i,0]=data[3][0]
            next_input[i,1]=data[3][1] 
            R[i]=data[2]
            A[i]=data[1]
        q_output=target_network(next_input)
        max_q, _=torch.max(q_output,dim=1)
        y_T=R+gamma*max_q
        for i in range(learning_epoch):
            optimizer.zero_grad()
            main_output=main_network(input)
            indexed_output = torch.zeros(batch_size)
            for j in range(batch_size):
                indexed_output[j] = main_output[j, A[j].long()]
            loss=mse(indexed_output,y_T)
            loss.backward(retain_graph=True)
            optimizer.step()
            print(f"Epoch: {i+1}, Loss: {loss.item()}")
    target_network.load_state_dict(main_network.state_dict())
    x,y=0,0
    dataset=[]
    for i in range(dataset_size):
        pv_x,pv_y=x,y
        q_value=main_network(torch.tensor([x,y],dtype=torch.float32))
        cur_dir = torch.argmax(q_value)
        prob_list=np.zeros(5)
        for i in range(5):
            if(i==cur_dir):
                prob_list[i]=1-epsilon*4/5
            else:
                prob_list[i]=epsilon/5
        cur_dir=random.choices(range(5), weights=prob_list)[0]
        r=state_list[pv_x][pv_y].reward[cur_dir]
        x,y=next_state(x,y,cur_dir)
        dataset.append([[pv_x,pv_y],cur_dir,r,[x,y]])
    print("________________________________________")
    q_value=main_network(torch.tensor([0,0],dtype=torch.float32))
    policy=torch.argmax(q_value)
    print(f"epoch {main_e} {q_value} {policy}")
    

# greed policy
for i in range(grid_width):
    for j in range(grid_height):
        q_value=main_network(torch.tensor([i,j],dtype=torch.float32))
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