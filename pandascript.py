
# import gym
# import panda_gym
# import time

# env = gym.make('PandaReach-v2', render=True)

# obs = env.reset()
# done = False
# while not done:
#     time.sleep(0.05)
#     action = env.action_space.sample() # random action
#     obs, reward, done, info = env.step(action)

# input()

# env.close()



import zarr

f = "data/panda_garbage-2.zarr"
z =zarr.open(f,path = "meta")
