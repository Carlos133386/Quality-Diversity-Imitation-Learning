import gym
import numpy as np
import torch

from brax.envs import env as brax_env
from brax import jumpy as jp


class TotalReward(brax_env.Wrapper):
    def reset(self, rng: jp.ndarray) -> brax_env.State:
        state = self.env.reset(rng)
        state.info['total_reward'] = jp.zeros(self.env.batch_size)
        state.info['traj_length'] = jp.zeros(self.env.batch_size)
        return state

    def step(self, state: brax_env.State, action: jp.ndarray) -> brax_env.State:
        nstate = self.env.step(state, action)
        if 'total_reward' in nstate.info:
            total_rew = nstate.info['total_reward']
            total_rew += nstate.reward
            state.info.update(total_reward=total_rew)
        if 'traj_length' in nstate.info:
            t = nstate.info['traj_length']
            t += jp.ones(self.env.batch_size)
            state.info.update(traj_length=t)
        return nstate

