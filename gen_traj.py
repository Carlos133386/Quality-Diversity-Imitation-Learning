#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path


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
import pdb

from ribs.archives._elite import Elite, EliteBatch
from ribs.archives._archive_base import readonly

device = torch.device('cuda')

normalize_obs = True
normalize_rewards = True

def config_env(env_name='ant', seed=1111):
    # params to config
    
    # env_name = 'humanoid'
    # env_name='ant'
    # env_name='walker2d'
    clip_obs_rew=False
    if env_name == 'humanoid':
        clip_obs_rew = True
    
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
            'clip_obs_rew': clip_obs_rew
    })



    # now lets load in a saved archive dataframe and scheduler
    # change this to be your own checkpoint path
    archive_path = f'experiments_1_best_elite/IL_ppga_{env_name}_expert/1111/checkpoints/cp_00002000/archive_df_00002000.pkl'
    scheduler_path = f'experiments_1_best_elite/IL_ppga_{env_name}_expert/1111/checkpoints/cp_00002000/scheduler_00002000.pkl'
    with open(archive_path, 'rb') as f:
        archive_df = pickle.load(f)
    with open(scheduler_path, 'rb') as f:
        scheduler = pickle.load(f)


    # create the environment
    env = make_vec_env_brax(env_cfg)

    return env, scheduler, actor_cfg, env_cfg


def get_best_elite(scheduler, actor_cfg):
    best_elite = scheduler.archive.best_elite
    print(f'Loading agent with reward {best_elite.objective} and measures {best_elite.measures}')
    agent = Actor(obs_shape=actor_cfg.obs_shape[0], action_shape=actor_cfg.action_shape, normalize_obs=normalize_obs, normalize_returns=normalize_rewards).deserialize(best_elite.solution).to(device)
    if actor_cfg.normalize_obs:
        norm = best_elite.metadata['obs_normalizer']
        if isinstance(norm, dict):
            agent.obs_normalizer.load_state_dict(norm)
        else:
            agent.obs_normalizer = norm
    return agent, best_elite.measures

def euclidean_dist(point, centroids):
    dist = np.sqrt(np.sum((point - centroids) ** 2, axis=1))
    return dist

def sample_top_k_diverse_elites(archive, topk=100, num_demo=4):
    """Randomly samples elites from the archive.

    Currently, this sampling is done uniformly at random. Furthermore, each
    sample is done independently, so elites may be repeated in the sample.
    Additional sampling methods may be supported in the future.

    Since :namedtuple:`EliteBatch` is a namedtuple, the result can be
    unpacked (here we show how to ignore some of the fields)::

        solution_batch, objective_batch, measures_batch, *_ = \\
            archive.sample_elites(32)

    Or the fields may be accessed by name::

        elite = sample_top_k_diverse_elites(archive, topk=100, num_demo=4) 
        elite.solution_batch
        elite.objective_batch
        ...

    Args:
        n (int): Number of elites to sample.
    Returns:
        EliteBatch: A batch of good and diverse elites selected from the archive.
    Raises:
        IndexError: The archive is empty.
    """
    if archive.empty:
        raise IndexError("No elements in archive.")

    ord_indices = np.arange(archive._num_occupied)
    occupied_indices = archive._occupied_indices[ord_indices]
    archive._solution_arr = archive._solution_arr[occupied_indices]
    archive._objective_arr = archive._objective_arr[occupied_indices]
    archive._measures_arr = archive._measures_arr[occupied_indices]
    archive._metadata_arr = archive._metadata_arr[occupied_indices]

    sorted_indices=np.argsort(archive._objective_arr)[::-1]
    topk_indices = sorted_indices[:topk]
    
    # Farthest Point Sampling
    #https://medium.com/@konyakinsergey/farthest-point-sampling-for-k-means-clustering-23a6dfc2dfb1
    data = archive._measures_arr[topk_indices]
    initial_centroid_idx = 0 # set the best elite as the initial centroid 
    # initial_centroid_idx = np.random.randint(topk) # Get random data point index
    initial_centroid = data[initial_centroid_idx]           # Retrieve the data point associated with that index
    centroids = [initial_centroid]                          # Put the data point into a list with centroid locations
    
    centroid_indices = [initial_centroid_idx]
    num_centorids = num_demo
    for i in range(1, num_centorids):
        distances = []

        # 3. Repeat 1 and 2 for each data point
        for x in data:
            # 1. Find distances between a point and all centroids
            dists = euclidean_dist(x, centroids)
            # 2. Save the min distance 
            distances.append(np.min(dists))

        # 4. Append the point with maximum distance; 
        # this will be the new farthest centroid
        max_idx = np.argmax(distances)
        centroids.append(data[max_idx])
        centroid_indices.append(max_idx)

    centroids = np.array(centroids)
    selected_indices = topk_indices[centroid_indices]
    
    elites = []
    for i in range(num_demo):
        selected_indices_ = np.asarray([selected_indices[i]])
        elite =  EliteBatch(
                        readonly(archive._solution_arr[selected_indices_]),
                        readonly(archive._objective_arr[selected_indices_]),
                        readonly(archive._measures_arr[selected_indices_]),
                        readonly(selected_indices_),
                        readonly(archive._metadata_arr[selected_indices_]),
                    )
        elites.append(elite)
    return elites, selected_indices, archive._measures_arr, archive._measures_arr[topk_indices]

def get_good_and_diverse_elite(elite, actor_cfg):
    print(f'Loading agent with reward {elite.objective_batch[0]} and measures {elite.measures_batch[0]}')
    agent = Actor(obs_shape=actor_cfg.obs_shape[0], action_shape=actor_cfg.action_shape, normalize_obs=normalize_obs, normalize_returns=normalize_rewards).deserialize(elite.solution_batch.flatten()).to(device)
    if actor_cfg.normalize_obs:
        norm = elite.metadata_batch[0]['obs_normalizer']
        if isinstance(norm, dict):
            agent.obs_normalizer.load_state_dict(norm)
        else:
            agent.obs_normalizer = norm
    return agent, elite.measures_batch[0]

def get_random_elite(scheduler, actor_cfg):
    elite = scheduler.archive.sample_elites(1)
    print(f'Loading agent with reward {elite.objective_batch[0]} and measures {elite.measures_batch[0]}')
    agent = Actor(obs_shape=actor_cfg.obs_shape[0], action_shape=actor_cfg.action_shape, normalize_obs=normalize_obs, normalize_returns=normalize_rewards).deserialize(elite.solution_batch.flatten()).to(device)
    if actor_cfg.normalize_obs:
        norm = elite.metadata_batch[0]['obs_normalizer']
        if isinstance(norm, dict):
            agent.obs_normalizer.load_state_dict(norm)
        else:
            agent.obs_normalizer = norm
    return agent, elite.measures_batch[0]


def gen_1_traj(env, agent, actor_cfg, env_cfg, render=False, deterministic=True):
    if actor_cfg.normalize_obs:
        obs_mean, obs_var = agent.obs_normalizer.obs_rms.mean, agent.obs_normalizer.obs_rms.var
        # print(f'{obs_mean=}, {obs_var=}')

    obs = env.reset()
    rollout = [env.unwrapped._state]
    total_reward = 0
    sum_measures = torch.zeros(env_cfg.num_dims).to(device)
    done = False
    eps_states = []
    eps_actions = []
    eps_rewards = []
    eps_measures = []

    # steps = 0
    reward = 0
    # eps_return = 0
    eps_length = 0
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
            sum_measures += info['measures']
            rollout.append(env.unwrapped._state)
            total_reward += rew
            eps_length += 1
            
        eps_states.append(obs.cpu().numpy())
        eps_actions.append(act.cpu().numpy())
        eps_rewards.append(reward)
        eps_measures.append(info['measures'].cpu().numpy())
    eps_states = np.array(eps_states)
    eps_actions = np.array(eps_actions)
    eps_rewards = np.array(eps_rewards)
    eps_measures = np.array(eps_measures)
   
    eps_return = total_reward.detach().cpu().numpy()
    

    if render:
        i = HTML(html.render(env.unwrapped._env.sys, [s.qp for s in rollout]))
        display(i)
        print(f'{total_reward=}')
        print(f' Rollout length: {len(rollout)}')
        sum_measures /= len(rollout)
        print(f'Measures: {sum_measures.cpu().numpy()}')
   
    return eps_states, eps_actions, eps_rewards, eps_measures, eps_return, eps_length
    

def gen_multi_trajs(agent_type='random', num_demo=10, env_name='ant', 
                    topk=100):
    print(env_name, '='*100)
    env, scheduler, actor_cfg, env_cfg = config_env(env_name)

    traj_root=f'trajs_{agent_type}_elite_with_measures'
    if agent_type=='good_and_diverse':
        traj_root += f'_top{topk}'
    os.makedirs(traj_root,exist_ok=True)

    os.makedirs(f'{traj_root}/{num_demo}episodes', exist_ok=True)

    lengths = []
    rewards = []
    returns = []
    states = []
    actions = []
    measures = []
    demonstrator_measures = []
    demonstrator_returns = []

    i = 0
    if agent_type == 'good_and_diverse':
        elites, selected_indices, full_occupied_measures, topk_occupied_measures = \
            sample_top_k_diverse_elites(scheduler.archive, topk=topk, num_demo=num_demo)
    for i in range(num_demo):
        if agent_type == 'best':
            agent, demonstrator_measure = get_best_elite(scheduler, actor_cfg)
        if agent_type == 'random':
            agent, demonstrator_measure = get_random_elite(scheduler, actor_cfg)
        if agent_type == 'good_and_diverse':
            elite = elites[i]
            agent, demonstrator_measure = get_good_and_diverse_elite(elite, actor_cfg)
        
        eps_states, eps_actions, eps_rewards, eps_measures, eps_return, eps_length = \
            gen_1_traj(env, agent, actor_cfg, env_cfg)
        states.append(eps_states)
        actions.append(eps_actions)
        rewards.append(eps_rewards)
        measures.append(eps_measures)
        returns.append(eps_return)
        lengths.append(eps_length)
        demonstrator_measures.append(demonstrator_measure)
        demonstrator_returns.append(elite.objective_batch[0])
        i +=1
        eps_measures_avg = eps_measures[:eps_length, ].sum(axis=0) / eps_length
        # print(env_name, 'Episode', i, '==========================')
        # print(i, 'eps_return', eps_return)
        # print(i, 'eps_length', eps_length)
        # print(i, 'demonstrator_measures', demonstrator_measure)
        # print(i, 'eps_measures avg', eps_measures_avg)
        
   
    states = np.concatenate(states, axis=0)
    actions = np.concatenate(actions, axis=0)
    measures = np.concatenate(measures, axis=0)
    rewards = np.concatenate(rewards, axis=0)
    returns = np.array(returns)
    lengths = np.array(lengths)
    demonstrator_measures = np.stack(demonstrator_measures, axis=0)
    demonstrator_returns = np.array(demonstrator_returns)

    
    print('states', states.shape)
    print('actions', actions.shape)
    print('measures', measures.shape)
    print('lengths', lengths)
    print('returns', returns )
    print('demonstrator returns', demonstrator_returns)
    print('demonstrator measures', demonstrator_measures.shape)
    print('demonstrator indices', selected_indices)


    traj = {}
    traj['states'] = states
    traj['actions'] = actions
    traj['rewards'] = rewards
    traj['measures'] = measures
    traj['returns'] = returns
    traj['lengths'] = lengths
    traj['demonstrator_returns'] = demonstrator_returns 
    traj['demonstrator_measures'] = demonstrator_measures
    if agent_type == 'good_and_diverse':
        traj['demonstrator_indices'] = selected_indices
        traj['full_occupied_measures'] = full_occupied_measures
        traj['topk_occupied_measures'] = topk_occupied_measures
 
    
    file_name = f'{traj_root}/{num_demo}episodes/trajs_ppga_{env_name}.pt'
    # pickle.dump(traj, open(file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

# topk=150
# topk=200
# topk=250
# topk=300
# topk=350
# topk=400
# topk=450
topk=500
# topk=600
# topk=700
# topk=800
# topk=900
# topk=1000
for num_demo in [4, 8, 16, 32, 64]:
    for env_name in ['ant', 'walker2d', 'humanoid']: # 
        # gen_multi_trajs(agent_type='best', num_demo=num_demo, env_name=env_name)
        # gen_multi_trajs(agent_type='random', num_demo=num_demo, env_name=env_name)
        gen_multi_trajs(agent_type='good_and_diverse', num_demo=num_demo, env_name=env_name, topk=topk)



