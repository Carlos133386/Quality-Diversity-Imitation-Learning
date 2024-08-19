import functools
import time

from IPython.display import HTML, Image
import gym

import brax

from attrdict import AttrDict
from envs import brax_custom
from brax.envs import to_torch
from jax.dlpack import to_dlpack
from envs.brax_custom.gpu_env import make_vec_env_brax
import torch
v = torch.ones(1, device='cuda')  # init torch cuda before jax


def brax_test(gym_env):
    # jit compile env.reset
    obs = gym_env.reset()

    # jit compile env.step
    action = torch.rand(gym_env.action_space.shape, device='cuda') * 2 - 1
    obs, reward, done, info = gym_env.step(action)

    before = time.time()

    steps = 1000
    for _ in range(steps):
        action = torch.rand(gym_env.action_space.shape, device='cuda') * 2 - 1
        obs, rewards, done, info = gym_env.step(action)

    duration = time.time() - before
    env_steps = gym_env.num_envs * steps
    print(f'time for {env_steps} steps: {duration:.2f}s ({int(env_steps / duration)} steps/sec)')


def two_brax_gyms():
    cfg = {'env_name': 'ant', 'seed': 0, 'env_batch_size': 4096}
    cfg = AttrDict(cfg)
    env1 = make_vec_env_brax(cfg)
    env2 = make_vec_env_brax(cfg)
    print('successfully spawned 2 brax env instances!')
    brax_test(env1)
    brax_test(env2)


if __name__ == '__main__':
    two_brax_gyms()
