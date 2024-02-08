import gymnasium as gym
import panda_gym
import time
import numpy as np

min_contact = [0.02, 0.02, 0.02]

env = gym.make("PandaPush-v3", render_mode="human")
observation, info = env.reset()


for _ in range(1000):
    current_position = observation["observation"][0:3]
    
    block_pos = observation["observation"][6:9]
    goal = observation["desired_goal"]
    contact = np.linalg.norm(block_pos - current_position)
    dir2goal = np.linalg.norm(block_pos - goal)

    if goal[1] >= block_pos[1]:
        min_contact[1] -= 0.01
    else:
        min_contact[1] += 0.01
    if goal[0] >= block_pos[0]:
        min_contact[0] -= 0.01
    else:
        min_contact[0] += 0.01

    print(np.linalg.norm(min_contact))
    input()

    while contact > np.linalg.norm(min_contact):

        block_pos[2] += 0.01
        action = 3.0 * (block_pos - current_position)
        
        print(contact)
        time.sleep(0.05)
        observation, reward, terminated, truncated, info = env.step(action)
        
        current_position = observation["observation"][0:3]
    
        block_pos = observation["observation"][6:9]
        goal = observation["desired_goal"]
        
        contact = np.linalg.norm(block_pos - current_position)
        
    print("WHILE END")
    #observation, reward, terminated, truncated, info = env.step(action)
    action = 3.0 * (goal - current_position)
    observation, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.05)
    if terminated or truncated:
        print("term",terminated)
        print("trunc",truncated)
        time.sleep(1)
        observation, info = env.reset()

env.close()