from __future__ import print_function

import argparse
import  gymnasium as gym
import time
import numpy as np
import os
from datetime import datetime
import json
import pygame
import glob
import copy
import pandas as pd


def append_df():
    # Collect all dataframes into one
    path = './original_data/dataset.pkl'
    # Find all the files with the .pkl extension in the current directory
    filenames = glob.glob('./original_data/*.pkl')

    # Create an empty DataFrame to store the results
    df_final = pd.DataFrame()

    # Iterate over the list of filenames
    for filename in filenames:
        # Load the DataFrame from the file
        df = pd.read_pickle(filename)
        
        # Append the DataFrame to the final DataFrame
        df_final = pd.concat([df_final, df], ignore_index=True)
    
    # Check if older data is available
    if os.path.isfile(path):
        # Read old data and append new data
        df = pd.read_pickle(path)
        df_final = pd.concat([df, df_final], ignore_index=True)
    
    # Delete the initial files
    for filename in filenames:
        os.remove(filename)

    # Save final df
    df_final.to_pickle(path)

def total_rewards(reward_list):
    # Save episode rewards
    path = './original_data/rewards/rewards.pkl'
    df_new = pd.DataFrame (rewards_dict.items())
    # Check existing data
    if os.path.isfile(path):
        df = pd.read_pickle(path)
        df = pd.concat([df_new, df], ignore_index=True)
    else:
        df = df_new
    # Save
    df.to_pickle(path)
    

def register_input(a):
    global quit, restart
    # Car Controls
    if pygame.key.get_pressed()[pygame.K_LEFT]:
        a = 2
    elif pygame.key.get_pressed()[pygame.K_RIGHT]:
        a = 1
    elif pygame.key.get_pressed()[pygame.K_UP]:
        a = 3
    elif pygame.key.get_pressed()[pygame.K_DOWN]:
        a = 4
    else:
        a = 0

    # Restart and Quit
    if pygame.key.get_pressed()[pygame.K_RETURN]:
        restart = True
    if pygame.key.get_pressed()[pygame.K_ESCAPE]:
        quit = True

    return a

if __name__ == "__main__":
    
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n' , "--num_episodes", type=int, default=10)
    args = parser.parse_args()

    # Create dictionary structure
    episode_samples = {
        "state": [],
        "action": [],
        "speed" : [],
    }
    env = gym.make('CarRacing-v2', continuous=False, render_mode="human")
    env.reset()

    a = 0
    rewards_dict = {}
    
    episode_rewards = []
    # Episode loop
    i=0
    while i < args.num_episodes:
        i += 1
        episode_samples["state"] = []
        episode_samples["action"] = []
        episode_samples["speed"] = []
        state = env.reset(seed=1)
        rewards = 0
        restart = False
        quit = False
        # State loop
        while True:
            # Save state before action has been taken
            episode_samples["state"].append(state)            # state has shape (60, 60, 3), original: (96, 96, 3)
            # Get action and give it to environment
            a = register_input(a)
            state, r, terminated, truncated, info = env.step(a)
            # time.sleep(0.1)
            # Record episode info
            speed = np.sqrt(np.square(env.car.hull.linearVelocity[0])+ np.square(env.car.hull.linearVelocity[1]))
            episode_samples["speed"].append(speed)
            episode_samples["action"].append(a)      
            rewards += r
            env.render()
            if terminated or truncated or restart or quit: 
                rewards_dict[i] = rewards
                print("Reward: ", rewards)
                break

        if quit:
            # If esc pressed, save previous data and exit
            break