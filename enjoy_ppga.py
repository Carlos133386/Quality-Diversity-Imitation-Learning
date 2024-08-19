#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
from pathlib import Path

# project_root = os.path.join(str(Path.home()), 'PPGA')
# os.chdir(project_root)
# get_ipython().run_line_magic('pwd', '# should be PPGA root dir')


# In[ ]:


import pickle

import numpy as np
from attrdict import AttrDict
from RL.ppo import *
from utils.utilities import log
from envs.brax_custom.brax_env import make_vec_env_brax
from models.actor_critic import Actor, PGAMEActor
from pandas import DataFrame

from IPython.display import HTML, Image
from IPython.display import display
from brax.io import html, image


# In[ ]:


# params to config
device = torch.device('cuda')
env_name = 'humanoid'
seed = 1111
normalize_obs = True
normalize_rewards = True
# non-configurable params
obs_shapes = {
    'humanoid': (227,),
    'ant': (87,),
    'halfcheetah': (18,),
    'walker2d': (17,)
}
action_shapes = {
    'humanoid': (17,),
    'ant': (8,),
    'halfcheetah': (6,),
    'walker2d': (6,)
}

# define the final config objects
actor_cfg = AttrDict({
        'obs_shape': obs_shapes[env_name],
        'action_shape': action_shapes[env_name],
        'normalize_obs': normalize_obs,
        'normalize_rewards': normalize_rewards,
})
env_cfg = AttrDict({
        'env_name': env_name,
        'env_batch_size': None,
        'num_dims': 2 if not 'ant' in env_name else 4,
        'envs_per_model': 1,
        'seed': seed,
        'num_envs': 1,
        'clip_obs_rew': False
})


# In[ ]:


# now lets load in a saved archive dataframe and scheduler
# change this to be your own checkpoint path
archive_path = 'experiments/paper_ppga_humanoid/1111/checkpoints/cp_00002000/archive_df_00002000.pkl'
scheduler_path = 'experiments/paper_ppga_humanoid/1111/checkpoints/cp_00002000/scheduler_00002000.pkl'
with open(archive_path, 'rb') as f:
    archive_df = pickle.load(f)
with open(scheduler_path, 'rb') as f:
    scheduler = pickle.load(f)


# In[ ]:


# create the environment
env = make_vec_env_brax(env_cfg)


# In[ ]:


def get_best_elite():
    best_elite = scheduler.archive.best_elite
    print(f'Loading agent with reward {best_elite.objective} and measures {best_elite.measures}')
    agent = Actor(obs_shape=actor_cfg.obs_shape[0], action_shape=actor_cfg.action_shape, normalize_obs=normalize_obs, normalize_returns=normalize_rewards).deserialize(best_elite.solution).to(device)
    if actor_cfg.normalize_obs:
        norm = best_elite.metadata['obs_normalizer']
        if isinstance(norm, dict):
            agent.obs_normalizer.load_state_dict(norm)
        else:
            agent.obs_normalizer = norm
    return agent


# In[ ]:


def get_random_elite():
    elite = scheduler.archive.sample_elites(1)
    print(f'Loading agent with reward {elite.objective_batch[0]} and measures {elite.measures_batch[0]}')
    agent = Actor(obs_shape=actor_cfg.obs_shape[0], action_shape=actor_cfg.action_shape, normalize_obs=normalize_obs, normalize_returns=normalize_rewards).deserialize(elite.solution_batch.flatten()).to(device)
    if actor_cfg.normalize_obs:
        norm = elite.metadata_batch[0]['obs_normalizer']
        if isinstance(norm, dict):
            agent.obs_normalizer.load_state_dict(norm)
        else:
            agent.obs_normalizer = norm
    return agent


# In[ ]:


def enjoy_brax(agent, render=True, deterministic=True):
    if actor_cfg.normalize_obs:
        obs_mean, obs_var = agent.obs_normalizer.obs_rms.mean, agent.obs_normalizer.obs_rms.var
        print(f'{obs_mean=}, {obs_var=}')

    obs = env.reset()
    rollout = [env.unwrapped._state]
    total_reward = 0
    measures = torch.zeros(env_cfg.num_dims).to(device)
    done = False
    while not done:
        with torch.no_grad():
            obs = obs.unsqueeze(dim=0).to(device)
            if actor_cfg.normalize_obs:
                obs = (obs - obs_mean) / torch.sqrt(obs_var + 1e-8)

            if deterministic:
                act = agent.actor_mean(obs)
            else:
                act, _, _ = agent.get_action(obs)
            act = act.squeeze()
            obs, rew, done, info = env.step(act.cpu())
            measures += info['measures']
            rollout.append(env.unwrapped._state)
            total_reward += rew
    if render:
        i = HTML(html.render(env.unwrapped._env.sys, [s.qp for s in rollout]))
        display(i)
        print(f'{total_reward=}')
        print(f' Rollout length: {len(rollout)}')
        measures /= len(rollout)
        print(f'Measures: {measures.cpu().numpy()}')
    return total_reward.detach().cpu().numpy()


# In[ ]:


agent = get_best_elite()
enjoy_brax(agent, render=True, deterministic=True)


# In[ ]:




