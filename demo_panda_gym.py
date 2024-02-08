import numpy as np
import click
from diffusion_policy.common.replay_buffer import ReplayBuffer
import gym
import panda_gym
#from panda_gym.envs.panda_tasks.panda_reach import PandaReachEnv 
from panda_gym.envs.panda_tasks.panda_robodiff import PandaReachDiffEnv 
import time
import cv2
import matplotlib.pyplot as plt
# env = gym.make("PandaReach-v3", render_mode="human")
# observation, info = env.reset()

# for _ in range(1000):
#     current_position = observation["observation"][0:3]
#     desired_position = observation["desired_goal"][0:3]
#     action = 5.0 * (desired_position - current_position)
#     observation, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated:
#         observation, info = env.reset()

# env.close()


#@click.command()
#@click.option('-o', '--output', required=True)
#@click.option('-rs', '--render_size', default=96, type=int)
#@click.option('-hz', '--control_hz', default=10, type=int)
def main(output="data/panda_garbage-3.zarr"):
    """
    Collect demonstration for the Push-T task.
    
    Usage: python demo_pusht.py -o data/pusht_demo.zarr
    
    This script is compatible with both Linux and MacOS.panda_garbage
    Hover mouse close to the blue circle to start.
    Push the T block into the green area. 
    The episode will automatically terminate if the task is succeeded.
    Press "Q" to exit.
    Press "R" to retry.
    Hold "Space" to pause.
    """
    
    # create replay buffer in read-write mode
    replay_buffer = ReplayBuffer.create_from_path(output, mode='a')

    #env:PandaReachEnv = gym.make('PandaReach-v2', render=True) #type annotation for pylance
    env:PandaReachDiffEnv = gym.make('PandaDiff-v0', render=True) #type annotation for pylance
    #info = env._get_info()
    #env = PandaReachEnv()

    observation = env.reset()

    episode = list()
    for _ in range(50000):
        
        seed = replay_buffer.n_episodes
        #print(f'starting seed {seed}')
        env.seed(seed)

        current_position = observation["observation"][0:3]
        desired_position = observation["desired_goal"][0:3]
        act = 5.0 * (desired_position - current_position)
        
        observation, reward, done, info = env.step(act)
    

        # img = env.render(mode='rgb_array',
        #                     width = 120, 
        #                     height= 120,
        #                     target_position = None,
        #                     distance = 0.75,
        #                     yaw = 45, #45
        #                     pitch= -45,
        #                     roll = 0
        #                     )
            
        # check image channels--will use later to confirm bgr/rgb type
        # plt.figure()
        # plt.imshow(img[:,:,0])
        # plt.figure()
        # plt.imshow(img[:,:,1])
        # plt.figure()
        # plt.imshow(img[:,:,2])
        # plt.show()
        
        ##show image for debugging
        # img = observation['image']
        # print(img.shape)
        # plt.figure()
        # plt.imshow(img)
        # plt.show()
        # time.sleep(0.1)


        # state = np.concatenate([current_position, desired_position])

        data = {
            'image': observation['image'],
            'action': np.float32(act),
            'observation': np.float32(observation['observation']),
            'achieved_goal': np.float32(observation['achieved_goal']),
            'desired_goal': np.float32(observation['desired_goal'])
        }

        episode.append(data)

        if done:
            print(len(episode))
            data_dict=dict()
            for key in episode[0].keys():
                data_dict[key] = np.stack(
                    [x[key] for x in episode])
            replay_buffer.add_episode(data_dict, compressors='disk')
            print(f'saved seed {seed}')
            obs = env.reset()
            episode = list()
        # if truncated:
        #     obs, info = env.reset()



if __name__ == "__main__":
    main()
