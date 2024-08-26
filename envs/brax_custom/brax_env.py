import functools
import gym
import brax

from envs import brax_custom
from brax.envs import to_torch
from jax.dlpack import to_dlpack

import torch
v = torch.ones(1, device='cuda')  # init torch cuda before jax

_to_custom_env = {
    'ant': {'custom_env_name': 'brax_custom-ant-v0',
            'kwargs': {
                'clip_actions': (-1, 1),
            }},
    'humanoid': {'custom_env_name': 'brax_custom-humanoid-v0',
                 'kwargs': {
                     'clip_actions': (-1, 1),
                 }},
    'walker2d': {'custom_env_name': 'brax-custom-walker2d-v0',
                 'kwargs': {
                     'clip_actions': (-1, 1),
                 }},
    'halfcheetah': {'custom_env_name': 'brax-custom-halfcheetah-v0',
                    'kwargs': {
                        'clip_actions': (-1, 1),
                    }},
    'hopper': {'custom_env_name': 'brax-custom-hopper-v0',
                    'kwargs': {
                        'clip_actions': (-1, 1),
                    }},
    'humanoidstandup': {'custom_env_name': 'brax_custom-humanoidstandup-v0',
                 'kwargs': {
                     'clip_actions': (-1, 1),
                 }},
}


def make_vec_env_brax(cfg):
    entry_point = functools.partial(brax_custom.create_gym_env, env_name=cfg.env_name)
    brax_env_name = _to_custom_env[cfg.env_name]['custom_env_name']
    if brax_env_name not in gym.envs.registry.env_specs:
        gym.register(brax_env_name, entry_point=entry_point)

    kwargs = _to_custom_env[cfg.env_name]['kwargs']
    if cfg.clip_obs_rew:
        kwargs['clip_obs'] = (-10, 10)
        kwargs['clip_rewards'] = (-10, 10)
    vec_env = gym.make(_to_custom_env[cfg.env_name]['custom_env_name'], batch_size=cfg.env_batch_size, seed=cfg.seed,
                       **kwargs)
    vec_env = to_torch.JaxToTorchWrapper(vec_env, device='cuda')

    return vec_env
