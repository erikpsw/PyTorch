import gymnasium as gym
import pygame

from sdc_wrapper import SDC_Wrapper

class ControlStatus:
    """
    Class to keep track of key presses while recording demonstrations.
    """
    def __init__(self):
        self.stop = False
        self.save = False
        self.quit = False
        self.steer = 0.0
        self.accelerate = 0.0
        self.brake = 0.0

def drive():
    """
    Function to drive in the environment.

    The controls are:
    arrow keys:         control the car; steer left, steer right, gas, brake
    ESC:                quit and close
    SPACE:              restart on a new track
    """
    env = SDC_Wrapper(gym.make('CarRacing-v2', render_mode='human'))
    status = ControlStatus()
    total_reward = 0.0

    status.quit = False

    while not status.quit:
        # get an observation from the environment
        observation, _ = env.reset()
        status = ControlStatus()
        while True:
            status=1
            status.update()

            # submit the users' action to the environment and get the reward
            # for that step as well as the new observation (status)
            observation, reward, done, trunc, info = env.step([status.steer,
                                                        status.accelerate,
                                                        status.brake])
            print(info)
            total_reward += reward

            if status.stop or status.save or status.quit:
                break

        status.stop = False
        status.save = False

    env.close()

if __name__ == "__main__":
    drive()