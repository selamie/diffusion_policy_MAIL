import gym
import gym.spaces
import panda_gym
from panda_gym.envs.panda_tasks.panda_reach import PandaReachEnv

import numpy as np

from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.tasks.reach import Reach
from panda_gym.pybullet import PyBullet
import gym.utils.seeding


from typing import Any, Dict, Optional, Tuple, Union


class PandaReachDiffEnv(RobotTaskEnv):

    def __init__(self, render_size=120, render: bool = False, reward_type: str = "dense", control_type: str = "ee") -> None:
        self.sim = PyBullet(render=render)
        self.robot = Panda(self.sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        self.task = Reach(self.sim, distance_threshold=0.005, reward_type=reward_type, get_ee_position=self.robot.get_ee_position)
        self.render_size = render_size

        obs = self.reset()  # required for init; seed can be changed later
        observation_shape = obs["observation"].shape
        achieved_goal_shape = obs["achieved_goal"].shape
        desired_goal_shape = obs["achieved_goal"].shape
        super().__init__(self.robot, self.task)
        self.observation_space = gym.spaces.Dict({
            'observation':gym.spaces.Box(-10.0, 10.0, shape=observation_shape, dtype=np.float32),
            'desired_goal':gym.spaces.Box(-10.0, 10.0, shape=achieved_goal_shape, dtype=np.float32),
            'achieved_goal':gym.spaces.Box(-10.0, 10.0, shape=desired_goal_shape, dtype=np.float32),
            'cam1':gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(3,render_size,render_size),
                    dtype=np.float32),
            'cam2':gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(3,render_size,render_size),
                    dtype=np.float32)
        }
        )
        
        self.render_cache=None
        self.done = False #only bc idk what's going on in the original

        

    def _get_obs(self) -> Dict[str, np.ndarray]:
        

        img1= self.sim.render(mode = 'rgb_array',
                            width = self.render_size, 
                            height= self.render_size,
                            target_position = None,
                            distance = 1.0, #0.75
                            yaw = 45, #45
                            pitch= -45,
                            roll = 0)
      
        img1 = np.delete(img1,3,axis=2)
        
        img2 = self.sim.render(mode = 'rgb_array',
                            width = self.render_size, 
                            height= self.render_size,
                            target_position = np.array([-0.5,0.25,0]),
                            distance = 1.0,
                            yaw = 225, #45
                            pitch= -20,
                            roll = 0)
        
        img2 = np.delete(img2,3,axis=2)

        if 1 == self.task.is_success(self.task.get_achieved_goal(),self.task.get_goal()):
              self.done=True

        robot_obs = self.robot.get_obs()  # robot state
        task_obs = self.task.get_obs()  # object position, velocity, etc...
        observation = np.concatenate([robot_obs, task_obs])
        #task_obs is empty for this array so observation IS agent_pos
        #has shape 6 but can try "turning off" velocity in next env layer if u want
        achieved_goal = self.task.get_achieved_goal()
        
        return {
            "observation": observation,
            "desired_goal": self.task.get_goal(),
            "achieved_goal": achieved_goal,
            "cam1": img1,
            "cam2": img2
        }
    
    def step(self,action):
        obs, reward, done, info = super().step(action)
        if self.done: 
            done = True
            self.done = False
            return obs, reward, done, info
        else:
            return obs, reward, done, info
    
