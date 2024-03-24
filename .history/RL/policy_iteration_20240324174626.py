import taichi as ti
import numpy as np

ti.init(ti.cpu)
pixel_size=60
grid_width=5
grid_height=5
width=pixel_size*grid_width
height=pixel_size*grid_height
step=1/grid_width
gamma=0.9
punishment=-10
epochs=30

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
        self.value=0
        self.policy=0
    
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
#初始化奖励
v_list=[]
for i in range(grid_width):
    tmp=[]
    for j in range(grid_height):
        obj=state_list[i][j]
        x=obj.x
        y=obj.y
        tmp.append(obj.value)
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
    v_list.append(tmp)
    
is_stable=False # 策略是否稳定
epoch=0

#policy iteration
while not is_stable:
    is_stable=True
    print(f'epoch {epoch}')
    epoch+=1
    
    # policy evaluation
    for t in range(epochs):
        # print(f"evaluation {t} {v_list[0]}")
        for i in range(grid_width):
            for j in range(grid_height):
                q_list=[]
                obj=state_list[i][j]
                dir=obj.policy
                nx,ny=next_state(i,j,dir)
                obj.value=obj.reward[dir]+(gamma*v_list[nx][ny])
        for i in range(grid_width):
            for j in range(grid_height):
                v_list[i][j]=state_list[i][j].value # v_list 为上一轮的value
                
    # policy improvement
    for i in range(grid_width):
        for j in range(grid_height):
            q_list=[]
            obj=state_list[i][j]
            for k in range(5):
                nx,ny=next_state(i,j,k)
                q_list.append(obj.reward[k]+(gamma*v_list[nx][ny]))
            dir = q_list.index(max(q_list))
            if(dir!=obj.policy):
                is_stable=False
            obj.policy=dir
            

# print(v_list)
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