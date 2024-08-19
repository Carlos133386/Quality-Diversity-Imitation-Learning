import sys
import torch
from envs.brax_custom.brax_env import make_vec_env_brax
import pdb
import argparse
from attrdict import AttrDict
from distutils.util import strtobool

def parse_args():
    parser = argparse.ArgumentParser()
    # PPO params
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='ant')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--torch_deterministic", type=lambda x: bool(strtobool(x)), default=False, nargs="?",
                        const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--use_wandb", default=False, type=lambda x: bool(strtobool(x)),
                        help='Use weights and biases to track the exp')
    parser.add_argument('--wandb_run_name', type=str, default='ppo_ant')
    parser.add_argument('--wandb_group', type=str)
    parser.add_argument('--wandb_project', type=str, default='PPGA')

    # args for brax
    parser.add_argument('--env_batch_size', default=1, type=int, help='Number of parallel environments to run')

    # ppo hyperparams
    parser.add_argument('--report_interval', type=int, default=5, help='Log objective results every N updates')
    parser.add_argument('--rollout_length', type=int, default=2048,
                        help='the number of steps to run in each environment per policy rollout')
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--anneal_lr', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help='Toggle learning rate annealing for policy and value networks')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for rewards')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='Lambda discount used for general advantage est')
    parser.add_argument('--num_minibatches', type=int, default=32)
    parser.add_argument('--update_epochs', type=int, default=10, help='The K epochs to update the policy')
    parser.add_argument("--norm_adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip_coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip_vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--clip_value_coef", type=float, default=0.2,
                        help="value clipping coefficient")
    parser.add_argument("--entropy_coef", type=float, default=0.0,
                        help="coefficient of the entropy")
    parser.add_argument("--vf_coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target_kl", type=float, default=None,
                        help="the target KL divergence threshold")
    parser.add_argument('--normalize_obs', type=lambda x: bool(strtobool(x)), default=False,
                        help='Normalize observations across a batch using running mean and stddev')
    parser.add_argument('--normalize_returns', type=lambda x: bool(strtobool(x)), default=False,
                        help='Normalize returns across a batch using running mean and stddev')
    parser.add_argument('--value_bootstrap', type=lambda x: bool(strtobool(x)), default=False,
                        help='Use bootstrap value estimates')
    parser.add_argument('--weight_decay', type=float, default=None, help='Apply L2 weight regularization to the NNs')
    parser.add_argument('--clip_obs_rew', type=lambda x: bool(strtobool(x)), default=False, help='Clip obs and rewards b/w -10 and 10')

    # QD Params
    parser.add_argument("--num_emitters", type=int, default=1, help="Number of parallel"
                                                                    " CMA-ES instances exploring the archive")
    parser.add_argument('--grid_size', type=int, default=10, help='Number of cells per archive dimension')
    parser.add_argument("--num_dims", type=int, default=1, help="Dimensionality of measures")
    parser.add_argument("--popsize", type=int, default=1,
                        help="Branching factor for each step of MEGA i.e. the number of branching solutions from the current solution point")
    parser.add_argument('--log_arch_freq', type=int, default=10,
                        help='Frequency in num iterations at which we checkpoint the archive')
    parser.add_argument('--save_scheduler', type=lambda x: bool(strtobool(x)), default=True,
                        help='Choose whether or not to save the scheduler during checkpointing. If the archive is too big,'
                             'it may be impractical to save both the scheduler and the archive_df. However, you cannot later restart from '
                             'a scheduler checkpoint and instead will have to restart from an archive_df checkpoint, which may impact the performance of the run.')
    parser.add_argument('--load_scheduler_from_cp', type=str, default=None,
                        help='Load an existing QD scheduler from a checkpoint path')
    parser.add_argument('--load_archive_from_cp', type=str, default=None,
                        help='Load an existing archive from a checkpoint path. This can be used as an alternative to loading the scheduler if save_scheduler'
                             'was disabled and only the archive df checkpoint is available. However, this can affect the performance of the run. Cannot be used together with save_scheduler')
    parser.add_argument('--total_iterations', type=int, default=100,
                        help='Number of iterations to run the entire dqd-rl loop')
    parser.add_argument('--dqd_algorithm', type=str, choices=['cma_mega_adam', 'cma_maega'],
                        help='Which DQD algorithm should be running in the outer loop')
    parser.add_argument('--expdir', type=str, help='Experiment results directory')
    parser.add_argument('--save_heatmaps', type=lambda x: bool(strtobool(x)), default=True,
                        help='Save the archive heatmaps. Only applies to archives with <= 2 measures')
    parser.add_argument('--use_surrogate_archive', type=lambda x: bool(strtobool(x)), default=False,
                        help="Use a surrogate archive at a higher resolution to get a better gradient signal for DQD")
    parser.add_argument('--sigma0', type=float, default=1.0,
                        help='Initial standard deviation parameter for the covariance matrix used in NES methods')
    parser.add_argument('--restart_rule', type=str, choices=['basic', 'no_improvement'])
    parser.add_argument('--calc_gradient_iters', type=int,
                        help='Number of iters to run PPO when estimating the objective-measure gradients (N1)')
    parser.add_argument('--move_mean_iters', type=int,
                        help='Number of iterations to run PPO when moving the mean solution point (N2)')
    parser.add_argument('--archive_lr', type=float, help='Archive learning rate for MAEGA')
    parser.add_argument('--threshold_min', type=float, default=0.0,
                        help='Min objective threshold for adding new solutions to the archive')
    parser.add_argument('--take_archive_snapshots', type=lambda x: bool(strtobool(x)), default=False,
                        help='Log the objective scores in every cell in the archive every log_freq iterations. Useful for pretty visualizations')
    parser.add_argument('--adaptive_stddev', type=lambda x: bool(strtobool(x)), default=True,
                        help='If False, the log stddev parameter in the actor will be reset on each QD iteration. Can potentially help exploration but may lose performance')

    args = parser.parse_args()
    cfg = AttrDict(vars(args))
    return cfg

if __name__ == '__main__':
    cfg = parse_args()
    cfg.num_emitters = 1

    vec_env = make_vec_env_brax(cfg)
    # pdb.set_trace()
    cfg.obs_shape = vec_env.single_observation_space.shape
    cfg.action_shape = vec_env.single_action_space.shape
    obs_dim = cfg.obs_shape[0]
    action_dim = cfg.action_shape[0]
    print(obs_dim, action_dim)
