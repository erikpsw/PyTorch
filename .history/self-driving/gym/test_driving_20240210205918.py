import gymnasium as gym
import pygame
import matplotlib.pyplot as plt
from sdc_wrapper import SDC_Wrapper

class ControlStatus:
    def __init__(self):
        self.stop = False
        self.save = False
        self.quit = False
        self.steer = 0.0
        self.accelerate = 0.0
        self.brake = 0.0

def drive():
    frame=0
    env = SDC_Wrapper(gym.make('CarRacing-v2', render_mode='human'))
    status = ControlStatus()
    total_reward = 0.0

    status.quit = False
    pygame.init()
    while not status.quit:
        observation, _ = env.reset()
        status = ControlStatus()
        while True:
            frame+=1
            status.steer=0.0
            status.accelerate=0.2
            
            observation, reward, done, trunc, info = env.step([status.steer,
                                                        status.accelerate,
                                                        status.brake])
            # observation 为96x96 RGB 图像
            # done terminated 返回是否终止
            # trunc truncated 截止 比如到了时间上限
            
            print(reward)
            # plt.imshow(observation)
            total_reward += reward
            
            if status.stop or status.save or status.quit:
                break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    status.quit=True

        status.stop = False
        status.save = False

    env.close()
    # plt.show()
    #

if __name__ == "__main__":
    drive()