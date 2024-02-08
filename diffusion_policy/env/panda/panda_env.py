import gym
import panda_gym
from panda_gym.envs.panda_tasks.panda_robodiff import PandaReachDiffEnv
import numpy as np

#prob need to ultimately rewrite this but...? 

class PandaEnv(PandaReachDiffEnv):
    def _init_(self, render_size):
        super.__init__()
        self.render_cache = None

    def _get_obs(self):
        obs = super()._get_obs()

        img = obs['image']
        img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0)
        ##^literally just so their evaluator is happy...
    
        #replace with a reshaped image
        obs['image'] = img_obs
        
        self.render_cache = img

        return obs
    
    def render(self, mode):
        assert mode == 'rgb_array'

        if self.render_cache is None:
            self._get_obs()
        
        return self.render_cache


