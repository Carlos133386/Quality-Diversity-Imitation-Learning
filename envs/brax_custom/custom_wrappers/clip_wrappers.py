from brax.envs import env as brax_env
from brax import jumpy as jp
from brax.envs import State


class ActionClipWrapper(brax_env.Wrapper):
    def __init__(self, env: brax_env.Env, a_min: float, a_max: float):
        super().__init__(env)
        self.a_min = jp.array(a_min)
        self.a_max = jp.array(a_max)

    def reset(self, rng: jp.ndarray) -> brax_env.State:
        state = self.env.reset(rng)
        return state

    def step(self, state: brax_env.State, action: jp.ndarray) -> brax_env.State:
        action = jp.clip(action, self.a_min, self.a_max)
        nstate = self.env.step(state, action)
        return nstate


class ObservationClipWrapper(brax_env.Wrapper):
    def __init__(self, env: brax_env.Env, obs_min, obs_max):
        super().__init__(env)
        self.obs_min = jp.array(obs_min)
        self.obs_max = jp.array(obs_max)

    def reset(self, rng: jp.ndarray) -> brax_env.State:
        state = self.env.reset(rng)
        obs = state.obs
        clipped_obs = jp.clip(obs, self.obs_min, self.obs_max)
        state = state.replace(obs=clipped_obs)
        return state

    def step(self, state: State, action: jp.ndarray) -> State:
        nstate = self.env.step(state, action)
        obs = nstate.obs
        clipped_obs = jp.clip(obs, self.obs_min, self.obs_max)
        nstate = nstate.replace(obs=clipped_obs)
        return nstate


class RewardClipWrapper(brax_env.Wrapper):
    def __init__(self, env: brax_env.Env, rew_min, rew_max):
        super().__init__(env)
        self.rew_min = jp.array(rew_min)
        self.rew_max = jp.array(rew_max)

    def reset(self, rng: jp.ndarray) -> State:
        state = self.env.reset(rng)
        rew = state.reward
        clipped_rew = jp.clip(rew, self.rew_min, self.rew_max)
        state = state.replace(reward=clipped_rew)
        return state

    def step(self, state: State, action: jp.ndarray) -> State:
        nstate = self.env.step(state, action)
        rew = nstate.reward
        clipped_rew = jp.clip(rew, self.rew_min, self.rew_max)
        nstate = nstate.replace(reward=clipped_rew)
        return nstate
