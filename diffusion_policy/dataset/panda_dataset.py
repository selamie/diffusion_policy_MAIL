from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class PandaDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            #zarr_path, keys=['img', 'state', 'action'])
            zarr_path, keys=['image', 'observation', 'action','achieved_goal','desired_goal'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        #THISS produces paramater_dict (!!)
        data = {
            'action': self.replay_buffer['action'],
            'observation': self.replay_buffer['observation'],
            'desired_goal': self.replay_buffer['desired_goal'],
            'achieved_goal': self.replay_buffer['achieved_goal'],
            'image':self.replay_buffer['image']
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        #agent_pos = sample['observation'][0:3].astype(np.float32) # (agent_posx2, block_posex3)
        #task_pos = sample['desired_goal'][0:3].astype(np.float32)
        observation = sample['observation']
        desired_goal = sample['desired_goal']
        achieved_goal = sample['achieved_goal']
        image = np.moveaxis(sample['image'],-1,1)/255
        #image = sample['image']
        data = {
            'obs': {
                'image': image, # T, 3, 96, 96
                'observation' : observation,
                'desired_goal' : desired_goal,
                'achieved_goal' : achieved_goal
            },
            'action': sample['action'].astype(np.float32) # T, 2
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def test():
    import os
    zarr_path = os.path.expanduser('~/diffusion_policy/data/panda_garbage-2.zarr')
    dataset = PandaDataset(zarr_path, horizon=16)

#     from matplotlib import pyplot as plt
#     normalizer = dataset.get_normalizer()
#     nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
#     diff = np.diff(nactions, axis=0)
#     dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)

