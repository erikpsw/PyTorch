import gymnasium as gym
import pygame

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
    
    env = SDC_Wrapper(gym.make('CarRacing-v2', render_mode='human'))
    status = ControlStatus()
    total_reward = 0.0

    status.quit = False
    pygame.init()
    while not status.quit:
        # get an observation from the environment
        observation, _ = env.reset()
        status = ControlStatus()
        while True:
            status.steer=1.0
            status.accelerate=1.0
            
            observation, reward, done, trunc, info = env.step([status.steer,
                                                        status.accelerate,
                                                        status.brake])
            print(info)
            total_reward += reward

            if status.stop or status.save or status.quit:
                break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

        status.stop = False
        status.save = False

    env.close()

if __name__ == "__main__":
    drive()