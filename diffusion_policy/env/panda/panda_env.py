import gym
import panda_gym
from panda_robodiff import PandaReachDiffEnv
import numpy as np

class PandaEnv(PandaReachDiffEnv):
    def _init_(self, render_size):
        super.__init__()
        self.render_cache = None

    def _get_obs(self):
        obs = super()._get_obs()

        img1 = obs['cam1']
        img1 = np.moveaxis(img1.astype(np.float32) / 255, -1, 0)
        img2 = obs['cam2']
        img2 = np.moveaxis(img2.astype(np.float32) / 255, -1, 0)
        #replace with a reshaped image
        obs['cam1'] = img1
        obs['cam2'] = img2

        self.render_cache = img2

        return obs
    
    def render(self, mode):
        assert mode == 'rgb_array'

        if self.render_cache is None:
            self._get_obs()
        
        return self.render_cache


