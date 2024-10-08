
import os
import os.path as osp
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch import autograd
import numpy as np
import argparse
from envs.brax_custom.brax_env import make_vec_env_brax
from distutils.util import strtobool
import os
import os.path as osp
from torch import nn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from algorithm.data_loader import ExpertDataset
from torch.utils.data import DataLoader

from attrdict import AttrDict
from torchvision.utils import make_grid
from torch.utils.data import Subset
import pdb

from math import ceil, exp, sqrt
from typing import Dict, Optional, Tuple


def process(tensor, normalize=False, range=None, scale_each=False):
    """Make a grid of images.
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    return tensor 

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    if np.isnan(batch_mean).any():
        print('nan value occures in batch_mean, convert to numbers')
        batch_mean = np.nan_to_num(batch_mean)
    if np.isnan(mean).any():
        print('nan value occures in mean, convert to numbers')
        mean = np.nan_to_num(mean)
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        if np.isnan(x).any():
            print('nan value occures in input x, convert to numbers')
            x = np.nan_to_num(x)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

class RMS(object):
    """running mean and std """
    def __init__(self, device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape).to(device)
        self.S = torch.ones(shape).to(device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs +
                 torch.square(delta) * self.n * bs /
                 (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S


def load_sa_data(args, return_next_state=True):
    traj_root = args.demo_dir
    traj_file = f'{traj_root}/trajs_ppga_{args.env_name}.pt'
    print(f'Loading data: {traj_file}')
    dataset = ExpertDataset(file_name=traj_file, num_trajectories=args.num_demo, train=True, 
                            train_test_split=1.0, return_next_state=return_next_state)
    dataloader = DataLoader(dataset, batch_size=args.gail_batchsize, shuffle=False, num_workers=1, drop_last=True)
    return dataset, dataloader

#### ICM model
# Inverse Dynamics model
class InverseModel(nn.Module):

    def __init__(self, input_dim, action_dim,
                 hidden_dim):

        super(InverseModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = action_dim
        self.hidden = hidden_dim

        # Inverse Model architecture
        self.linear_1 = nn.Linear(in_features=self.input_dim*2, out_features=self.hidden)
        self.linear_2 = nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.output = nn.Linear(in_features=self.hidden, out_features=self.output_dim)

        # Leaky relu activation
        self.tanh_1 = nn.Tanh()
        self.tanh_2 = nn.Tanh()

        # Output Activation
        # self.softmax = nn.Softmax()

        # Initialize the weights using xavier initialization
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, state, next_state):

        # Concatenate the state and the next state
        input = torch.cat([state, next_state], dim=-1)
        x = self.linear_1(input)
        x = self.tanh_1(x)
        x = self.linear_2(x)
        x = self.tanh_2(x)
        x = self.output(x)
        #output = self.softmax(x)
        return x


# Forward Dynamics Model
class ForwardDynamicsModel(nn.Module):

    def __init__(self, obs_dim, action_dim,
                 hidden_dim, output_dim=None):

        super(ForwardDynamicsModel, self).__init__()

        self.input_dim = obs_dim+action_dim
        if output_dim == None:
            self.output_dim = obs_dim
        else:
            self.output_dim= output_dim
        self.hidden = hidden_dim

        # Forward Model Architecture
        self.linear_1 = nn.Linear(in_features=self.input_dim, out_features=self.hidden)
        self.linear_2 = nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.output = nn.Linear(in_features=self.hidden, out_features=self.output_dim)

        # Leaky Relu activation
        self.tanh_1 = nn.Tanh()
        self.tanh_2 = nn.Tanh()

        # Initialize the weights using xavier initialization
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, state, action):
        # Concatenate the state and the action
        # Note that the state in this case is the feature representation of the state
        input = torch.cat([state, action], dim=-1)
        x = self.linear_1(input)
        x = self.tanh_1(x)
        x = self.linear_2(x)
        x = self.tanh_2(x)
        output = self.output(x)

        return output






# GAIL and VAIL

# GAIL discriminator
class GAILdiscriminator(nn.Module):

    def __init__(self, input_dim, 
                 hidden_dim, action_dim):

        super(GAILdiscriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 1
        self.hidden = hidden_dim
        self.action_dim = action_dim 

        # discriminator architecture
        self.linear_1 = nn.Linear(in_features=self.input_dim+self.action_dim, out_features=self.hidden)
        self.linear_2 = nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.output = nn.Linear(in_features=self.hidden, out_features=self.output_dim)

        # Leaky relu activation
        self.tanh_1 = nn.Tanh()
        self.tanh_2 = nn.Tanh()

        # Output Activation
        # self.softmax = nn.Softmax()

        # Initialize the weights using xavier initialization
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.tanh_1(x)
        x = self.linear_2(x)
        x = self.tanh_2(x)
        x = self.output(x)
        return x

class GAILdiscriminator_wo_a(nn.Module):

    def __init__(self, input_dim, 
                 hidden_dim, action_dim):

        super(GAILdiscriminator_wo_a, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 1
        self.hidden = hidden_dim
        self.action_dim = action_dim 

        # discriminator architecture
        self.linear_1 = nn.Linear(in_features=self.input_dim, out_features=self.hidden)
        self.linear_2 = nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.output = nn.Linear(in_features=self.hidden, out_features=self.output_dim)

        # Leaky relu activation
        self.tanh_1 = nn.Tanh()
        self.tanh_2 = nn.Tanh()

        # Output Activation
        # self.softmax = nn.Softmax()

        # Initialize the weights using xavier initialization
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, x):
        #x的最后一个维度只取前input_dim个
        x = x[..., :self.input_dim]
        
        x = self.linear_1(x)
        x = self.tanh_1(x)
        x = self.linear_2(x)
        x = self.tanh_2(x)
        x = self.output(x)
        return x
    
    
    
    
    
from math import ceil, exp, sqrt
from typing import Dict, Optional, Tuple
from torch import Tensor
class PWIL(object):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 expert_dataloader,
                 reward_scale=1.0,
                 reward_bandwidth_scale=1.0,
                 time_horizon=1000,
                 device='cuda:0',
                 ):

        self.device = device
        self.action_dim = action_dim
        self.expert_dataloader = expert_dataloader
        self.expert_memory = iter(self.expert_dataloader)
        self.time_horizon =  time_horizon
        self.data_scale, self.data_offset = self._calculate_normalisation_scale_offset(self._get_expert_atoms())  # Calculate normalisation parameters for the data
        self.reward_scale = reward_scale 
        self.reward_bandwidth = reward_bandwidth_scale * self.time_horizon / sqrt(obs_dim + action_dim)  # Reward function hyperparameters (based on α and β)
        self.reset()

    # Returns the scale and offset to normalise data based on mean and standard deviation
    def _calculate_normalisation_scale_offset(self, data: Tensor) -> Tuple[Tensor, Tensor]:
        inv_scale, offset = data.std(dim=0, keepdims=True), -data.mean(dim=0, keepdims=True)  # Calculate statistics over dataset
        inv_scale[inv_scale == 0] = 1  # Set (inverse) scale to 1 if feature is constant (no variance)
        return 1 / inv_scale, offset

    # Returns a tensor with a "row" (dim 0) deleted
    def _delete_row(self, data: Tensor, index: int) -> Tensor:
        return torch.cat([data[:index], data[index + 1:]], dim=0)

    def _get_expert_atoms(self, expert_state=None, expert_action=None) -> Tensor:
        if expert_state==None and expert_action==None:
            try:
                expert_batch = next(self.expert_memory)
            except:
                self.expert_memory = iter(self.expert_dataloader)
                expert_batch = next(self.expert_memory)
            expert_state, expert_action, expert_m = expert_batch
        return torch.cat([expert_state, expert_action], dim=1).to(self.device)

    def reset(self):
        self.expert_atoms = self.data_scale * (self._get_expert_atoms() + self.data_offset)  # Get and normalise the expert atoms
        self.expert_weights = torch.full((self.expert_atoms.shape[0], ), 1 / self.expert_atoms.shape[0]).to(self.device)

    # parallized version
    def calculate_intrinsic_reward(self, state, action, 
                                   use_original_reward=True, alpha=1e-8, reward_type=None):
        # Normalize agent atoms for the whole batch
        agent_atoms = torch.cat([state, action], dim=1)
        agent_atoms = self.data_scale * (agent_atoms + self.data_offset)  # Normalize agent atoms

        # Compute pairwise distances between agent atoms and expert atoms
        dists = torch.cdist(agent_atoms, self.expert_atoms)  # Shape: [batch_size, num_expert_atoms]

        # Initialize weights and costs
        weights = torch.full((state.shape[0],), 1 / self.time_horizon - 1e-6).to(self.device)
        costs = torch.zeros(state.shape[0], device=self.device)

        # Repeat the expert atoms and weights to match batch size
        expert_atoms = self.expert_atoms.clone()
        expert_weights = self.expert_weights.clone()

        # Iterate until all weights in the batch are zero
        while torch.any(weights > 0):
            # Find the closest expert atom for each agent atom
            closest_expert_idx = dists.argmin(dim=1)  # Shape: [batch_size]
            closest_dists = dists.gather(1, closest_expert_idx.unsqueeze(1)).squeeze(1)

            # Get expert weights of the closest atoms
            closest_expert_weights = expert_weights[closest_expert_idx]

            # Compute how much weight to subtract
            weight_to_subtract = torch.minimum(weights, closest_expert_weights)

            # Update the costs for each agent
            costs += weight_to_subtract * closest_dists

            # Update expert weights and batch weights
            expert_weights[closest_expert_idx] -= weight_to_subtract
            weights -= weight_to_subtract

            # Set weights of exhausted expert atoms to zero
            expert_mask = expert_weights > 0
            expert_atoms = expert_atoms[expert_mask]
            expert_weights = expert_weights[expert_mask]

            # Update distances by keeping only the remaining expert atoms
            dists = torch.cdist(agent_atoms, expert_atoms)

        # Calculate rewards for the batch
        rewards = self.reward_scale * torch.exp(-self.reward_bandwidth * costs)

        return rewards
    
    
    
# VAIL discriminator
class VAILdiscriminator(nn.Module):

    def __init__(self, input_dim, 
                 hidden_dim, action_dim):

        super(VAILdiscriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 1
        self.hidden = hidden_dim
        self.action_dim = action_dim 

        # Inverse Model architecture
        self.linear_1 = nn.Linear(in_features=self.input_dim+self.action_dim, out_features=self.hidden)
        self.linear_2 = nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.output = nn.Linear(in_features=int(self.hidden/2), out_features=self.output_dim)

        # Leaky relu activation
        self.tanh_1 = nn.Tanh()
        self.tanh_2 = nn.Tanh()

        # Output Activation
        self.sigmoid_output = nn.Sigmoid()

    def forward(self, x, mean_mode=True):
        x = self.linear_1(x)
        x = self.tanh_1(x)
        x = self.linear_2(x)
        x = self.tanh_2(x)
        
        parameters = x.squeeze(-1).squeeze(-1)

        # split the activations into means and standard deviations
        halfpoint = parameters.shape[-1] // 2
        mus, sigmas = parameters[:, :halfpoint], parameters[:, halfpoint:]
        sigmas = self.sigmoid_output(sigmas)  # sigmas are restricted to be from 0 to 1

        if not mean_mode:
            # sample point from gaussian distribution
            # this is for the discriminator
            out = (torch.randn_like(mus).to(x.device) * sigmas) + mus
        else:
            out = mus
                                
        out = self.output(out)

        return out, mus, sigmas

# ACGAIL discriminator
class ACGAILdiscriminator(nn.Module):

    def __init__(self, input_dim, hidden_dim, action_dim, measure_dim):

        super(ACGAILdiscriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden = hidden_dim
        self.action_dim = action_dim 
        self.measure_dim = measure_dim

        # discriminator architecture
        self.linear_1 = nn.Linear(in_features=self.input_dim+self.action_dim, out_features=self.hidden)
        self.linear_2 = nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.output_d = nn.Linear(in_features=self.hidden, out_features=1)
        self.output_m = nn.Linear(in_features=self.hidden, out_features=self.measure_dim)

        # Leaky relu activation
        self.tanh_1 = nn.Tanh()
        self.tanh_2 = nn.Tanh()

        # Output Activation

        # Initialize the weights using xavier initialization
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.xavier_uniform_(self.output_d.weight)
        nn.init.xavier_uniform_(self.output_m.weight)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.tanh_1(x)
        x = self.linear_2(x)
        x = self.tanh_2(x)
        d = self.output_d(x)
        m = self.output_m(x)
        return d, m
    
### GAIL 
class GAIL(object):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 device='cuda:0',
                 lr=3e-4,
                 ):


        self.device = device
        self.action_dim = action_dim
        
        self.discriminator = GAILdiscriminator(input_dim=obs_dim,  \
                                     hidden_dim=100, action_dim=self.action_dim).to(device)


        self.lr = lr
        self.intrinsic_reward_rms = RMS(device=self.device)

        self.gail_optim = optim.Adam(
                                [
                                    {'params': self.discriminator.parameters()}
                                ],
                                lr=self.lr
                                )
        
        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())
        self.ob_rms = RunningMeanStd(shape=())

    def compute_grad_pen(self,
                        expert_state,
                        expert_action,
                        policy_state,
                        policy_action,
                        lambda_=10):

       expert_data = torch.cat([expert_state, expert_action], dim=1)
       policy_data = torch.cat([policy_state, policy_action], dim=1)

       alpha = torch.rand_like(expert_data).to(expert_data.device)

       mixup_data = alpha * expert_data + (1 - alpha) * policy_data
       mixup_data.requires_grad = True

       disc = self.discriminator(mixup_data)
       ones = torch.ones(disc.size()).to(disc.device)
       grad = autograd.grad(
           outputs=disc,
           inputs=mixup_data,
           grad_outputs=ones,
           create_graph=True,
           retain_graph=True,
           only_inputs=True)[0]

       grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
       return grad_pen

    def feed_forward_generator(self,
                               b_obs, 
                               b_actions,
                               num_minibatches,
                               minibatch_size
                               ):

        batch_size = b_obs.shape[1]
        # if minibatch_size is None:
        #     minibatch_size = batch_size // num_minibatches
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            minibatch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = b_obs.view(-1, b_obs.shape[-1])[indices]
            actions_batch = b_actions.view(-1, b_actions.shape[-1])[indices]

            yield obs_batch, actions_batch


    def update(self, expert_loader, num_minibatches, b_obs, b_actions, obsfilt=None):
        policy_data_generator = self.feed_forward_generator(b_obs, b_actions, num_minibatches, 
                                                            expert_loader.batch_size)

        loss = 0
        n = 0
        # pdb.set_trace()
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            policy_state, policy_action = policy_batch[0], policy_batch[1]
            policy_d = self.discriminator(
                torch.cat([policy_state, policy_action], dim=1))

            expert_state, expert_action, _ = expert_batch
            if obsfilt is not None:
                expert_state = obsfilt(expert_state.numpy(), update=False)
            
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_d = self.discriminator(
                torch.cat([expert_state, expert_action], dim=1))

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits( 
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(expert_state, expert_action,
                                            policy_state, policy_action)

            loss += (gail_loss + grad_pen).item()
            # loss += gail_loss.item()
            n += 1

            self.gail_optim.zero_grad()
            (gail_loss + grad_pen).backward()
            # gail_loss.backward()
            self.gail_optim.step()
        return loss / n

    def calculate_intrinsic_reward(self, state, action, 
                       use_original_reward=True, alpha=1e-8, reward_type='log1-d'):
        with torch.no_grad():
            d = self.discriminator(torch.cat([state, action], dim=1))
            s = torch.sigmoid(d)  
            # solution from here: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/issues/204
            
            if reward_type == 'logd':
                reward =  torch.log(s + alpha)
            if reward_type == 'log1-d':
                reward =  - torch.log(1 - s + alpha)
            if self.returns is None:
                self.returns = reward.clone()

            if use_original_reward:
                return reward 
            else:
                return reward / np.sqrt(self.ret_rms.var[0] + alpha)
#archive_bonus_GAIL
class abGAIL(object):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 measure_dim,
                 device='cuda:0',
                 bonus_type='single_step_bonus',
                 lr=3e-4,
                 wo_a = False
                 ):


        self.device = device
        self.action_dim = action_dim
        self.bonus_type = bonus_type
        if wo_a:
            self.discriminator = GAILdiscriminator_wo_a(input_dim=obs_dim,  \
                                     hidden_dim=100, action_dim=self.action_dim).to(device)
        else:
            self.discriminator = GAILdiscriminator(input_dim=obs_dim,  \
                                     hidden_dim=100, action_dim=self.action_dim).to(device)


        self.lr = lr
        self.intrinsic_reward_rms = RMS(device=self.device)

        self.gail_optim = optim.Adam(
                                [
                                    {'params': self.discriminator.parameters()}
                                ],
                                lr=self.lr
                                )
        
        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())
        self.ob_rms = RunningMeanStd(shape=())
        self.env_info = {'obs_dim': obs_dim, 'action_dim': action_dim, 'measure_dim': measure_dim}
        self.single_step_archive = torch.ones([2]*measure_dim).to(self.device)#shape: 2*2*2*2...(measure_dim)
        
    def update_single_step_archive(self, single_step_measure):
        '''
        single_step_measure: tensor of shape (batch_size, measure_dim)
        e.g. [[1,1,0,0],[0,0,1,0]]
        '''
        indices = single_step_measure.t().long() 
        values = torch.ones(single_step_measure.size(0)).to(self.device)  

        
        self.single_step_archive.index_put_(tuple(indices), values, accumulate=True)
    def calculate_single_step_bonus(self, single_step_measure,k=5):
        '''
        archive_distribution: tensor of shape (2,2,2,2,...,2) 2^measure_dim
        single_step_measure: tensor of shape (batch_size, measure_dim)
        e.g. [[1,1,0,0],[0,0,1,0]]
        
        return: tensor of shape (batch_size,)
        '''
        archive_distribution = self.single_step_archive / self.single_step_archive.sum()
        
        indices = list(single_step_measure.long().t()) # measure_dim * batch_size
       
        prob = archive_distribution[indices] # shape: (batch_size,)
        
        bonus = 1/(1 + prob) # is it good?
        bonus = 0.5*torch.exp(-k * prob)

        return bonus

    def compute_grad_pen(self,
                        expert_state,
                        expert_action,
                        policy_state,
                        policy_action,
                        lambda_=10):

       expert_data = torch.cat([expert_state, expert_action], dim=1)
       policy_data = torch.cat([policy_state, policy_action], dim=1)

       alpha = torch.rand_like(expert_data).to(expert_data.device)

       mixup_data = alpha * expert_data + (1 - alpha) * policy_data
       mixup_data.requires_grad = True

       disc = self.discriminator(mixup_data)
       ones = torch.ones(disc.size()).to(disc.device)
       grad = autograd.grad(
           outputs=disc,
           inputs=mixup_data,
           grad_outputs=ones,
           create_graph=True,
           retain_graph=True,
           only_inputs=True)[0]

       grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
       return grad_pen

    def feed_forward_generator(self,
                               b_obs, 
                               b_actions,
                               num_minibatches,
                               minibatch_size
                               ):

        batch_size = b_obs.shape[1]
        # if minibatch_size is None:
        #     minibatch_size = batch_size // num_minibatches
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            minibatch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = b_obs.view(-1, b_obs.shape[-1])[indices]
            actions_batch = b_actions.view(-1, b_actions.shape[-1])[indices]

            yield obs_batch, actions_batch


    def update(self, expert_loader, num_minibatches, b_obs, b_actions, obsfilt=None):
        policy_data_generator = self.feed_forward_generator(b_obs, b_actions, num_minibatches, 
                                                            expert_loader.batch_size)

        loss = 0
        n = 0
        # pdb.set_trace()
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            policy_state, policy_action = policy_batch[0], policy_batch[1]
            policy_d = self.discriminator(
                torch.cat([policy_state, policy_action], dim=1))

            expert_state, expert_action, _ = expert_batch
            if obsfilt is not None:
                expert_state = obsfilt(expert_state.numpy(), update=False)
            
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_d = self.discriminator(
                torch.cat([expert_state, expert_action], dim=1))

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits( 
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(expert_state, expert_action,
                                            policy_state, policy_action)

            loss += (gail_loss + grad_pen).item()
            # loss += gail_loss.item()
            n += 1

            self.gail_optim.zero_grad()
            (gail_loss + grad_pen).backward()
            # gail_loss.backward()
            self.gail_optim.step()
        return loss / n

    def calculate_intrinsic_reward(self, state, action, measure, value=None,
                       use_original_reward=True, alpha=1e-8, reward_type='log1-d'):
        with torch.no_grad():
            d = self.discriminator(torch.cat([state, action], dim=1))
            s = torch.sigmoid(d)  
            # solution from here: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/issues/204
            bonus = 0
            if reward_type == 'logd':
                reward =  torch.log(s + alpha)
            if reward_type == 'log1-d':
                reward =  - torch.log(1 - s + alpha)
            if self.returns is None:
                self.returns = reward.clone()
            if self.bonus_type == 'single_step_bonus':
                bonus = self.calculate_single_step_bonus(measure)
                self.update_single_step_archive(measure)
            
            reward = reward.squeeze(1) + bonus

            if use_original_reward:
                return reward 
            else:
                return reward / np.sqrt(self.ret_rms.var[0] + alpha)
# measure conditioned GAIL
class mCondGAIL(object):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 measure_dim, 
                 device='cuda:0',
                 lr=3e-4,
                 wo_a = False
                 ):


        self.device = device
        self.action_dim = action_dim
        if wo_a:
            self.discriminator = GAILdiscriminator_wo_a(input_dim=obs_dim+measure_dim,  \
                                     hidden_dim=100, action_dim=self.action_dim).to(device)
        else:
            self.discriminator = GAILdiscriminator(input_dim=obs_dim+measure_dim,  \
                                     hidden_dim=100, action_dim=self.action_dim).to(device)


        self.lr = lr
        self.intrinsic_reward_rms = RMS(device=self.device)

        self.gail_optim = optim.Adam(
                                [
                                    {'params': self.discriminator.parameters()}
                                ],
                                lr=self.lr
                                )
        
        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())
        self.ob_rms = RunningMeanStd(shape=())

    def compute_grad_pen(self,
                        expert_state,
                        expert_action,
                        policy_state,
                        policy_action,
                        lambda_=10):

       expert_data = torch.cat([expert_state, expert_action], dim=1)
       policy_data = torch.cat([policy_state, policy_action], dim=1)

       alpha = torch.rand_like(expert_data).to(expert_data.device)

       mixup_data = alpha * expert_data + (1 - alpha) * policy_data
       mixup_data.requires_grad = True

       disc = self.discriminator(mixup_data)
       ones = torch.ones(disc.size()).to(disc.device)
       grad = autograd.grad(
           outputs=disc,
           inputs=mixup_data,
           grad_outputs=ones,
           create_graph=True,
           retain_graph=True,
           only_inputs=True)[0]

       grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
       return grad_pen

    def feed_forward_generator(self,
                               b_obs, 
                               b_actions,
                               b_measure,
                               num_minibatches,
                               minibatch_size=None):

        batch_size = b_obs.shape[1]
        if minibatch_size is None:
            minibatch_size = batch_size // num_minibatches
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            minibatch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = b_obs.view(-1, b_obs.shape[-1])[indices]
            actions_batch = b_actions.view(-1, b_actions.shape[-1])[indices]
            measure_batch = b_measure.view(-1, b_measure.shape[-1])[indices] 

            yield obs_batch, actions_batch, measure_batch


    def update(self, expert_loader, num_minibatches, b_obs, b_actions, b_measure, obsfilt=None):
        policy_data_generator = self.feed_forward_generator(b_obs, b_actions, b_measure, \
                                    num_minibatches, expert_loader.batch_size)

        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            policy_state, policy_action, policy_measure = policy_batch[0], policy_batch[1], policy_batch[2]
            policy_d = self.discriminator(
                torch.cat([policy_state, policy_measure, policy_action], dim=1))

            expert_state, expert_action, expert_measure = expert_batch
            if obsfilt is not None:
                expert_state = obsfilt(expert_state.numpy(), update=False)
            
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_measure = expert_measure.to(self.device)
            expert_d = self.discriminator(
                torch.cat([expert_state, expert_measure, expert_action], dim=1))

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits( 
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(torch.cat([expert_state, expert_measure], dim=1), 
                                             expert_action,
                                             torch.cat([policy_state, policy_measure], dim=1), 
                                             policy_action)

            loss += (gail_loss + grad_pen).item()
            # loss += gail_loss.item()
            n += 1

            self.gail_optim.zero_grad()
            (gail_loss + grad_pen).backward()
            # gail_loss.backward()
            self.gail_optim.step()
        return loss / n

    def calculate_intrinsic_reward(self, state, action, measure,
                       use_original_reward=True, alpha=1e-8, reward_type='log1-d'):
        with torch.no_grad():
            d = self.discriminator(torch.cat([state, measure, action], dim=1))
            s = torch.sigmoid(d)  
            # solution from here: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/issues/204
            
            if reward_type == 'logd':
                reward =  torch.log(s + alpha)
            if reward_type == 'log1-d':
                reward =  - torch.log(1 - s + alpha)
            if self.returns is None:
                self.returns = reward.clone()

            if use_original_reward:
                return reward 
            else:
                return reward / np.sqrt(self.ret_rms.var[0] + alpha)
            
            
            


# measure-targeted Auxiliary Classifier GAIL
class mACGAIL(object):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 measure_dim,
                 auxiliary_loss_fn='MSE',
                 bonus_type=None,
                 device='cuda:0',
                 lr=3e-4,
                 ):


        self.device = device
        self.action_dim = action_dim
        self.measure_dim = measure_dim
        self.auxiliary_loss_fn = auxiliary_loss_fn
        self.bonus_type = bonus_type
        if self.bonus_type == 'measure_entropy':
            rms = RMS(self.device)
            knn_rms = False
            knn_k = 500
            knn_avg = True
            knn_clip = 0.0
            self.pbe = PBE(rms, knn_clip, knn_k, knn_avg, knn_rms, self.device)
        if 'fitness_cond_measure_entropy' in self.bonus_type:
            knn_k = 500
            self.vcse = VCSE(knn_k)
        
        self.discriminator = ACGAILdiscriminator(input_dim=obs_dim, hidden_dim=100, 
                                                 action_dim=self.action_dim, measure_dim=self.measure_dim).to(device)


        self.lr = lr
        self.intrinsic_reward_rms = RMS(device=self.device)

        self.gail_optim = optim.Adam(
                                [
                                    {'params': self.discriminator.parameters()}
                                ],
                                lr=self.lr
                                )
        
        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())
        self.ob_rms = RunningMeanStd(shape=())

    def compute_grad_pen(self,
                        expert_state,
                        expert_action,
                        policy_state,
                        policy_action,
                        lambda_=10):

       expert_data = torch.cat([expert_state, expert_action], dim=1)
       policy_data = torch.cat([policy_state, policy_action], dim=1)

       alpha = torch.rand_like(expert_data).to(expert_data.device)

       mixup_data = alpha * expert_data + (1 - alpha) * policy_data
       mixup_data.requires_grad = True

       disc, _ = self.discriminator(mixup_data)
       ones = torch.ones(disc.size()).to(disc.device)
       grad = autograd.grad(
           outputs=disc,
           inputs=mixup_data,
           grad_outputs=ones,
           create_graph=True,
           retain_graph=True,
           only_inputs=True)[0]

       grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
       return grad_pen

    def feed_forward_generator(self,
                               b_obs, 
                               b_actions,
                               b_measure,
                               num_minibatches,
                               minibatch_size=None):

        batch_size = b_obs.shape[1]
        if minibatch_size is None:
            minibatch_size = batch_size // num_minibatches
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            minibatch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = b_obs.view(-1, b_obs.shape[-1])[indices]
            actions_batch = b_actions.view(-1, b_actions.shape[-1])[indices]
            measure_batch = b_measure.view(-1, b_measure.shape[-1])[indices] 

            yield obs_batch, actions_batch, measure_batch

    def update(self, expert_loader, num_minibatches, b_obs, b_actions, b_measure, obsfilt=None):
        policy_data_generator = self.feed_forward_generator(b_obs, b_actions, b_measure, 
                                                            num_minibatches, expert_loader.batch_size)

        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            policy_state, policy_action, policy_measure = policy_batch[0], policy_batch[1], policy_batch[2]
            policy_d, policy_m_pred = self.discriminator(
                torch.cat([policy_state, policy_action], dim=1))

            expert_state, expert_action, expert_measure = expert_batch
            if obsfilt is not None:
                expert_state = obsfilt(expert_state.numpy(), update=False)
            
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_measure = expert_measure.to(self.device)
            expert_d, expert_m_pred = self.discriminator(
                torch.cat([expert_state, expert_action], dim=1))

            expert_d_loss = F.binary_cross_entropy_with_logits(
                expert_d,torch.ones(expert_d.size()).to(self.device))
            if self.auxiliary_loss_fn == 'MSE':
                expert_m_loss = F.mse_loss(expert_m_pred, expert_measure)
            elif self.auxiliary_loss_fn == 'NLL':
                expert_m_loss = NLL_loss(expert_m_pred, expert_measure)
            else:
                expert_m_loss = 0

            expert_loss = expert_d_loss + expert_m_loss
            
            policy_d_loss = F.binary_cross_entropy_with_logits( 
                policy_d, torch.zeros(policy_d.size()).to(self.device))
            if self.auxiliary_loss_fn == 'MSE':
                policy_m_loss = F.mse_loss(policy_m_pred, policy_measure)
            elif self.auxiliary_loss_fn == 'NLL':
                policy_m_loss = NLL_loss(policy_m_pred, policy_measure)
            else:
                policy_m_loss = 0 

            policy_loss = policy_d_loss + policy_m_loss 

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(expert_state, expert_action,
                                            policy_state, policy_action)

            loss += (gail_loss + grad_pen).item()
            n += 1

            self.gail_optim.zero_grad()
            (gail_loss + grad_pen).backward()
            self.gail_optim.step()
        return loss / n

    def calculate_intrinsic_reward(self, state, action, measure, value=None,
                       use_original_reward=True, alpha=1e-8, reward_type='log1-d'):
        with torch.no_grad():
            d, m = self.discriminator(torch.cat([state, action], dim=1))
            s = torch.sigmoid(d)  
            # solution from here: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/issues/204
            
            if reward_type == 'logd':
                gail_reward =  torch.log(s + alpha)
            if reward_type == 'log1-d':
                gail_reward =  - torch.log(1 - s + alpha)
            
            gail_reward = gail_reward.squeeze(1)
            if self.bonus_type is not None:
                if self.bonus_type == 'measure_error':
                    bonus = F.mse_loss(m, measure, reduction='none').mean(dim=1)
                if self.bonus_type == 'measure_error_nll':
                    bonus = NLL_loss(m, measure, reduction='none').mean(dim=1)
                if self.bonus_type == 'measure_entropy':
                    bonus = self.pbe(measure).squeeze(1)
                if self.bonus_type == 'fitness_cond_measure_entropy' and value is not None:
                    bonus = self.vcse(measure, value)[0].squeeze(1)

                reward = gail_reward + bonus 
            else:
                reward = gail_reward

            if self.returns is None:
                self.returns = reward.clone()

            if use_original_reward:
                return reward 
            else:
                return reward / np.sqrt(self.ret_rms.var[0] + alpha)

# measure conditioned Auxiliary Classifier GAIL
class mCondACGAIL(object):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 measure_dim, 
                 auxiliary_loss_fn='MSE',
                 bonus_type=None,
                 device='cuda:0',
                 lr=3e-4,
                 ):


        self.device = device
        self.action_dim = action_dim
        self.auxiliary_loss_fn = auxiliary_loss_fn
        self.bonus_type = bonus_type
        if self.bonus_type == 'measure_entropy':
            rms = RMS(self.device)
            knn_rms = False
            knn_k = 500
            knn_avg = True
            knn_clip = 0.0
            self.pbe = PBE(rms, knn_clip, knn_k, knn_avg, knn_rms, self.device)
        if 'fitness_cond_measure_entropy' in self.bonus_type:
            knn_k = 500
            self.vcse = VCSE(knn_k)
        
        self.discriminator = ACGAILdiscriminator(input_dim=obs_dim+measure_dim,  \
                                     hidden_dim=100, action_dim=self.action_dim,
                                     measure_dim=measure_dim).to(device)


        self.lr = lr
        self.intrinsic_reward_rms = RMS(device=self.device)

        self.gail_optim = optim.Adam(
                                [
                                    {'params': self.discriminator.parameters()}
                                ],
                                lr=self.lr
                                )
        
        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())
        self.ob_rms = RunningMeanStd(shape=())

    def compute_grad_pen(self,
                        expert_state,
                        expert_action,
                        policy_state,
                        policy_action,
                        lambda_=10):

       expert_data = torch.cat([expert_state, expert_action], dim=1)
       policy_data = torch.cat([policy_state, policy_action], dim=1)

       alpha = torch.rand_like(expert_data).to(expert_data.device)

       mixup_data = alpha * expert_data + (1 - alpha) * policy_data
       mixup_data.requires_grad = True

       disc, _ = self.discriminator(mixup_data)
       ones = torch.ones(disc.size()).to(disc.device)
       grad = autograd.grad(
           outputs=disc,
           inputs=mixup_data,
           grad_outputs=ones,
           create_graph=True,
           retain_graph=True,
           only_inputs=True)[0]

       grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
       return grad_pen

    def feed_forward_generator(self,
                               b_obs, 
                               b_actions,
                               b_measure,
                               num_minibatches,
                               minibatch_size=None):

        batch_size = b_obs.shape[1]
        if minibatch_size is None:
            minibatch_size = batch_size // num_minibatches
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            minibatch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = b_obs.view(-1, b_obs.shape[-1])[indices]
            actions_batch = b_actions.view(-1, b_actions.shape[-1])[indices]
            measure_batch = b_measure.view(-1, b_measure.shape[-1])[indices] 

            yield obs_batch, actions_batch, measure_batch


    def update(self, expert_loader, num_minibatches, b_obs, b_actions, b_measure, obsfilt=None):
        policy_data_generator = self.feed_forward_generator(b_obs, b_actions, b_measure, \
                                    num_minibatches, expert_loader.batch_size)

        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            policy_state, policy_action, policy_measure = policy_batch[0], policy_batch[1], policy_batch[2]
            policy_d, policy_m_pred = self.discriminator(
                torch.cat([policy_state, policy_measure, policy_action], dim=1))

            expert_state, expert_action, expert_measure = expert_batch
            if obsfilt is not None:
                expert_state = obsfilt(expert_state.numpy(), update=False)
            
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_measure = expert_measure.to(self.device)
            expert_d, expert_m_pred = self.discriminator(
                torch.cat([expert_state, expert_measure, expert_action], dim=1))

            expert_d_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            if self.auxiliary_loss_fn == 'MSE':
                expert_m_loss = F.mse_loss(expert_m_pred, expert_measure)
            elif self.auxiliary_loss_fn == 'NLL':
                expert_m_loss = NLL_loss(expert_m_pred, expert_measure)
            else:
                expert_m_loss = 0

            expert_loss = expert_d_loss + expert_m_loss
            
            policy_d_loss = F.binary_cross_entropy_with_logits( 
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))
            if self.auxiliary_loss_fn == 'MSE':
                policy_m_loss = F.mse_loss(policy_m_pred, policy_measure)
            elif self.auxiliary_loss_fn == 'NLL':
                policy_m_loss = NLL_loss(policy_m_pred, policy_measure)
            else:
                policy_m_loss = 0

            policy_loss = policy_d_loss + policy_m_loss 

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(torch.cat([expert_state, expert_measure], dim=1), 
                                             expert_action,
                                             torch.cat([policy_state, policy_measure], dim=1), 
                                             policy_action)

            loss += (gail_loss + grad_pen).item()
            n += 1

            self.gail_optim.zero_grad()
            (gail_loss + grad_pen).backward()
            self.gail_optim.step()
        return loss / n

    def calculate_intrinsic_reward(self, state, action, measure, value=None,
                       use_original_reward=True, alpha=1e-8, reward_type='log1-d'):
        with torch.no_grad():
            d, m = self.discriminator(torch.cat([state, measure, action], dim=1))
            s = torch.sigmoid(d)  
            # solution from here: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/issues/204
            
            if reward_type == 'logd':
                gail_reward =  torch.log(s + alpha)
            if reward_type == 'log1-d':
                gail_reward =  - torch.log(1 - s + alpha)

            gail_reward = gail_reward.squeeze(1)
            if self.bonus_type is not None:
                if self.bonus_type == 'measure_error':
                    bonus = F.mse_loss(m, measure, reduction='none').mean(dim=1)
                if self.bonus_type == 'measure_error_nll':
                    bonus = NLL_loss(m, measure, reduction='none').mean(dim=1)
                if self.bonus_type == 'measure_entropy':
                    bonus = self.pbe(measure).squeeze(1)
                if self.bonus_type == 'fitness_cond_measure_entropy' and value is not None:
                    bonus = self.vcse(measure, value)[0].squeeze(1)

                reward = gail_reward + bonus 
            else:
                reward = gail_reward
            
            if self.returns is None:
                self.returns = reward.clone()

            if use_original_reward:
                return reward 
            else:
                return reward / np.sqrt(self.ret_rms.var[0] + alpha)
            
def NLL_loss(output1, target, eps=1e-12, reduction='mean'):
    output = F.softmax(output1, dim=1)
    l = -target * torch.log(output+eps)
    if reduction == 'mean':
        loss = (torch.sum(l)) / l.size(0)
    else:
        loss = l
    return loss 

# measure regularized GAIL
class mRegGAIL(object):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 measure_dim,
                 reg_loss_fn='MSE',
                 bonus_type='measure_error',
                 device='cuda:0',
                 lr=3e-4,
                 ):


        self.device = device
        self.action_dim = action_dim
        self.reg_loss_fn = reg_loss_fn 
        
        self.discriminator = GAILdiscriminator(input_dim=obs_dim,  \
                                     hidden_dim=100, action_dim=self.action_dim).to(device)
        self.measure_predict_model = ForwardDynamicsModel(obs_dim=obs_dim, action_dim=action_dim, \
                                                      hidden_dim=100, output_dim=measure_dim).to(device)
        self.bonus_type = bonus_type
        if self.bonus_type == 'measure_entropy':
            rms = RMS(self.device)
            knn_rms = False
            knn_k = 500
            knn_avg = True
            knn_clip = 0.0
            self.pbe = PBE(rms, knn_clip, knn_k, knn_avg, knn_rms, self.device)
        if 'fitness_cond_measure_entropy' in self.bonus_type:
            knn_k = 500
            self.vcse = VCSE(knn_k)

        self.lr = lr
        self.intrinsic_reward_rms = RMS(device=self.device)

        self.gail_optim = optim.Adam(
                                [
                                    {'params': self.discriminator.parameters()},
                                    {'params': self.measure_predict_model.parameters()}
                                ],
                                lr=self.lr
                                )
        
        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())
        self.ob_rms = RunningMeanStd(shape=())
        self.env_info = {'obs_dim': obs_dim, 'action_dim': action_dim, 'measure_dim': measure_dim}
        self.single_step_archive = torch.ones([2]*measure_dim).to(self.device)#shape: 2*2*2*2...(measure_dim)
        
    def update_single_step_archive(self, single_step_measure):
        '''
        single_step_measure: tensor of shape (batch_size, measure_dim)
        e.g. [[1,1,0,0],[0,0,1,0]]
        '''
        indices = single_step_measure.t().long()  # Transpose to get the indices in the correct format
        values = torch.ones(single_step_measure.size(0)).to(self.device)  # Create a tensor of ones with size batch_size

        # Use scatter_add_ to accumulate values in the archive
        self.single_step_archive.index_put_(tuple(indices), values, accumulate=True)
    def calculate_single_step_bonus(self, single_step_measure):
        '''
        archive_distribution: tensor of shape (2,2,2,2,...,2) 2^measure_dim
        single_step_measure: tensor of shape (batch_size, measure_dim)
        e.g. [[1,1,0,0],[0,0,1,0]]
        
        return: tensor of shape (batch_size,)
        '''
        archive_distribution = self.single_step_archive / self.single_step_archive.sum()
        
        indices = list(single_step_measure.long().t()) # measure_dim * batch_size
        
        prob = archive_distribution[indices] # shape: (batch_size,)
        
        bonus = 1/(1 + prob) # is it good?
        return bonus
        
        
        
        
        
    def compute_grad_pen(self,
                        expert_state,
                        expert_action,
                        policy_state,
                        policy_action,
                        lambda_=10):

       expert_data = torch.cat([expert_state, expert_action], dim=1)
       policy_data = torch.cat([policy_state, policy_action], dim=1)

       alpha = torch.rand_like(expert_data).to(expert_data.device)

       mixup_data = alpha * expert_data + (1 - alpha) * policy_data
       mixup_data.requires_grad = True

       disc = self.discriminator(mixup_data)
       ones = torch.ones(disc.size()).to(disc.device)
       grad = autograd.grad(
           outputs=disc,
           inputs=mixup_data,
           grad_outputs=ones,
           create_graph=True,
           retain_graph=True,
           only_inputs=True)[0]

       grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
       return grad_pen

    def feed_forward_generator(self,
                               b_obs, 
                               b_actions,
                               b_measure,
                               num_minibatches,
                               minibatch_size=None):

        batch_size = b_obs.shape[1]
        if minibatch_size is None:
            minibatch_size = batch_size // num_minibatches
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            minibatch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = b_obs.view(-1, b_obs.shape[-1])[indices]
            actions_batch = b_actions.view(-1, b_actions.shape[-1])[indices]
            measure_batch = b_measure.view(-1, b_measure.shape[-1])[indices]

            yield obs_batch, actions_batch, measure_batch


    def update(self, expert_loader, num_minibatches, b_obs, b_actions, b_measure, obsfilt=None):
        policy_data_generator = self.feed_forward_generator(b_obs, b_actions, b_measure, 
                                                            num_minibatches, expert_loader.batch_size)

        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            policy_state, policy_action, policy_measure = policy_batch[0], policy_batch[1], policy_batch[2]
            policy_d = self.discriminator(
                torch.cat([policy_state, policy_action], dim=1))
            pred_policy_measure = self.measure_predict_model(policy_state, policy_action)

            expert_state, expert_action, expert_measure = expert_batch
            if obsfilt is not None:
                expert_state = obsfilt(expert_state.numpy(), update=False)
            
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_measure = expert_measure.to(self.device)
            expert_d = self.discriminator(
                torch.cat([expert_state, expert_action], dim=1))
            pred_expert_measure = self.measure_predict_model(expert_state, expert_action)

            expert_d_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device)) 
            if self.reg_loss_fn == 'MSE':
                expert_m_loss = F.mse_loss(pred_expert_measure, expert_measure)
            if self.reg_loss_fn == 'NLL':
                expert_m_loss = NLL_loss(pred_expert_measure, expert_measure)
            expert_loss = expert_d_loss + expert_m_loss 

            policy_d_loss = F.binary_cross_entropy_with_logits( 
                policy_d,
                torch.zeros(policy_d.size()).to(self.device)) 
            
            if self.reg_loss_fn == 'MSE':
                policy_m_loss = F.mse_loss(pred_policy_measure, policy_measure)
            if self.reg_loss_fn == 'NLL':
                policy_m_loss = NLL_loss(pred_policy_measure, policy_measure)

            policy_loss = policy_d_loss + policy_m_loss 

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(expert_state, expert_action,
                                            policy_state, policy_action)

            loss += (gail_loss + grad_pen).item()
            # loss += gail_loss.item()
            n += 1

            self.gail_optim.zero_grad()
            (gail_loss + grad_pen).backward()
            # gail_loss.backward()
            self.gail_optim.step()
        
        return loss / n

    def calculate_intrinsic_reward(self, state, action, measure, value=None,
                       use_original_reward=True, alpha=1e-8, reward_type='log1-d'):
        with torch.no_grad():
            d = self.discriminator(torch.cat([state, action], dim=1))
            s = torch.sigmoid(d)  
            # solution from here: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/issues/204
            
            if reward_type == 'logd':
                gail_reward =  torch.log(s + alpha)
            if reward_type == 'log1-d':
                gail_reward =  - torch.log(1 - s + alpha)

            gail_reward = gail_reward.squeeze(1)
            bonus = 0
            if self.bonus_type == 'measure_error':
                pred_measure = self.measure_predict_model(state, action)
                bonus = F.mse_loss(pred_measure, measure, reduction='none').mean(dim=1)
            if self.bonus_type == 'measure_error_nll':
                bonus = NLL_loss(pred_measure, measure, reduction='none').mean(dim=1)
            if self.bonus_type == 'measure_entropy':
                bonus = self.pbe(measure).squeeze(1)
            if self.bonus_type == 'fitness_cond_measure_entropy' and value is not None:
                bonus = self.vcse(measure, value)[0].squeeze(1)
            if self.bonus_type == 'weighted_fitness_cond_measure_entropy' and value is not None:
                bonus_v, n_v,n_m, eps, state_norm, value_norm = self.vcse(measure, value)
                bonus_m = torch.digamma(n_m+1)/measure.shape[0] + torch.log(eps*2+0.00001)
                weight = 1/torch.abs(bonus_m)
                bonus = weight*bonus_v
                bonus = bonus.squeeze(1)
            if self.bonus_type == 'single_step_bonus':
                bonus = self.calculate_single_step_bonus(measure)
                self.update_single_step_archive(measure)

            reward = gail_reward + bonus 

            if self.returns is None:
                self.returns = reward.clone()

            if use_original_reward:
                return reward 
            else:
                return reward / np.sqrt(self.ret_rms.var[0] + alpha)

# measure conditioned and regularized GAIL
class mCondRegGAIL(object):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 measure_dim,
                 reg_loss_fn='MSE',
                 bonus_type='measure_error',
                 device='cuda:0',
                 lr=3e-4,
                 ):


        self.device = device
        self.action_dim = action_dim
        
        self.discriminator = GAILdiscriminator(input_dim=obs_dim+measure_dim,  \
                                     hidden_dim=100, action_dim=self.action_dim).to(device)
        self.measure_predict_model = ForwardDynamicsModel(obs_dim=obs_dim, action_dim=action_dim, \
                                                      hidden_dim=100, output_dim=measure_dim).to(device)
        self.bonus_type = bonus_type
        self.reg_loss_fn = reg_loss_fn
        if self.bonus_type == 'measure_entropy':
            rms = RMS(self.device)
            knn_rms = False
            knn_k = 500
            knn_avg = True
            knn_clip = 0.0
            self.pbe = PBE(rms, knn_clip, knn_k, knn_avg, knn_rms, self.device)
        if 'fitness_cond_measure_entropy' in self.bonus_type:
            knn_k = 500
            self.vcse = VCSE(knn_k)

        self.lr = lr
        self.intrinsic_reward_rms = RMS(device=self.device)

        self.gail_optim = optim.Adam(
                                [
                                    {'params': self.discriminator.parameters()},
                                    {'params': self.measure_predict_model.parameters()}
                                ],
                                lr=self.lr
                                )
        
        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())
        self.ob_rms = RunningMeanStd(shape=())
        self.env_info = {'obs_dim': obs_dim, 'action_dim': action_dim, 'measure_dim': measure_dim}
        self.single_step_archive = torch.ones([2]*measure_dim).to(self.device)#shape: 2*2*2*2...(measure_dim)
        
    def update_single_step_archive(self, single_step_measure):
        '''
        single_step_measure: tensor of shape (batch_size, measure_dim)
        e.g. [[1,1,0,0],[0,0,1,0]]
        '''
        indices = single_step_measure.t().long() 
        values = torch.ones(single_step_measure.size(0)).to(self.device)  

        
        self.single_step_archive.index_put_(tuple(indices), values, accumulate=True)
    def calculate_single_step_bonus(self, single_step_measure):
        '''
        archive_distribution: tensor of shape (2,2,2,2,...,2) 2^measure_dim
        single_step_measure: tensor of shape (batch_size, measure_dim)
        e.g. [[1,1,0,0],[0,0,1,0]]
        
        return: tensor of shape (batch_size,)
        '''
        archive_distribution = self.single_step_archive / self.single_step_archive.sum()
        
        indices = list(single_step_measure.long().t()) # measure_dim * batch_size
        
        prob = archive_distribution[indices] # shape: (batch_size,)
        
        bonus = 1/(1 + prob) # is it good?
        return bonus

    def compute_grad_pen(self,
                        expert_state,
                        expert_action,
                        policy_state,
                        policy_action,
                        lambda_=10):

       expert_data = torch.cat([expert_state, expert_action], dim=1)
       policy_data = torch.cat([policy_state, policy_action], dim=1)

       alpha = torch.rand_like(expert_data).to(expert_data.device)

       mixup_data = alpha * expert_data + (1 - alpha) * policy_data
       mixup_data.requires_grad = True

       disc = self.discriminator(mixup_data)
       ones = torch.ones(disc.size()).to(disc.device)
       grad = autograd.grad(
           outputs=disc,
           inputs=mixup_data,
           grad_outputs=ones,
           create_graph=True,
           retain_graph=True,
           only_inputs=True)[0]

       grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
       return grad_pen

    def feed_forward_generator(self,
                               b_obs, 
                               b_actions,
                               b_measure,
                               num_minibatches,
                               minibatch_size=None):

        batch_size = b_obs.shape[1]
        if minibatch_size is None:
            minibatch_size = batch_size // num_minibatches
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            minibatch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = b_obs.view(-1, b_obs.shape[-1])[indices]
            actions_batch = b_actions.view(-1, b_actions.shape[-1])[indices]
            measure_batch = b_measure.view(-1, b_measure.shape[-1])[indices]

            yield obs_batch, actions_batch, measure_batch


    def update(self, expert_loader, num_minibatches, b_obs, b_actions, b_measure, obsfilt=None):
        policy_data_generator = self.feed_forward_generator(b_obs, b_actions, b_measure, 
                                                            num_minibatches, expert_loader.batch_size)

        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            policy_state, policy_action, policy_measure = policy_batch[0], policy_batch[1], policy_batch[2]
            policy_d = self.discriminator(
                torch.cat([policy_state, policy_measure, policy_action], dim=1))
            pred_policy_measure = self.measure_predict_model(policy_state, policy_action)

            expert_state, expert_action, expert_measure = expert_batch
            if obsfilt is not None:
                expert_state = obsfilt(expert_state.numpy(), update=False)
            
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_measure = expert_measure.to(self.device)
            expert_d = self.discriminator(
                torch.cat([expert_state, expert_measure, expert_action], dim=1))
            pred_expert_measure = self.measure_predict_model(expert_state, expert_action)

            expert_d_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            if self.reg_loss_fn == 'MSE':
                expert_m_loss = F.mse_loss(pred_expert_measure, expert_measure)
            if self.reg_loss_fn == 'NLL':
                expert_m_loss = NLL_loss(pred_expert_measure, expert_measure)
            expert_loss = expert_d_loss + expert_m_loss 

            policy_d_loss = F.binary_cross_entropy_with_logits( 
                policy_d,
                torch.zeros(policy_d.size()).to(self.device)) 
            if self.reg_loss_fn == 'MSE':
                policy_m_loss = F.mse_loss(pred_policy_measure, policy_measure)
            if self.reg_loss_fn == 'NLL':
                policy_m_loss = NLL_loss(pred_policy_measure, policy_measure)
            policy_loss = policy_d_loss + policy_m_loss 

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(torch.cat([expert_state, expert_measure], dim=1), 
                                             expert_action,
                                             torch.cat([policy_state, policy_measure], dim=1), 
                                             policy_action)

            loss += (gail_loss + grad_pen).item()
            n += 1

            self.gail_optim.zero_grad()
            (gail_loss + grad_pen).backward()
            self.gail_optim.step()
        return loss / n

    def calculate_intrinsic_reward(self, state, action, measure, value=None,
                       use_original_reward=True, alpha=1e-8, reward_type='log1-d'):
        '''
        for humanoid:
        state: [3000,227] bs*state_dim
        action: [3000,17] bs*action_dim
        measure: [3000,2] bs*n_mdim
        
        
        '''
        
        with torch.no_grad():
            d = self.discriminator(torch.cat([state, measure, action], dim=1))
            s = torch.sigmoid(d)  
            # solution from here: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/issues/204
            
            if reward_type == 'logd':
                gail_reward =  torch.log(s + alpha)
            if reward_type == 'log1-d':
                gail_reward =  - torch.log(1 - s + alpha)
            bonus = 0
            gail_reward = gail_reward.squeeze(1)
            if self.bonus_type == 'measure_error':
                pred_measure = self.measure_predict_model(state, action)
                bonus = F.mse_loss(pred_measure, measure, reduction='none').mean(dim=1)
            if self.bonus_type == 'measure_error_nll':
                bonus = NLL_loss(pred_measure, measure, reduction='none').mean(dim=1)
            if self.bonus_type == 'measure_entropy':
                bonus = self.pbe(measure).squeeze(1)
            if self.bonus_type == 'fitness_cond_measure_entropy' and value is not None:
                bonus = self.vcse(measure, value)[0].squeeze(1)
            if self.bonus_type == 'weighted_fitness_cond_measure_entropy' and value is not None:
                bonus_v, n_v,n_m, eps, state_norm, value_norm = self.vcse(measure, value)
                bonus_m = torch.digamma(n_m+1)/measure.shape[0] + torch.log(eps*2+0.00001)
                weight = 1/torch.abs(bonus_m)
                bonus = weight*bonus_v
                bonus = bonus.squeeze(1)
            if self.bonus_type == 'single_step_bonus':
                bonus = self.calculate_single_step_bonus(measure)
                self.update_single_step_archive(measure)
    
            reward = gail_reward + bonus 

            if self.returns is None:
                self.returns = reward.clone()

            if use_original_reward:
                return reward 
            else:
                return reward / np.sqrt(self.ret_rms.var[0] + alpha)

### VAIL
class VAIL(object):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 i_c=0.5,
                 device='cuda:0',
                 lr=3e-4,
                 ):

        self.device = device
        self.i_c = i_c
        self.action_dim = action_dim 
        self.discriminator = VAILdiscriminator(input_dim=obs_dim,  \
                                     hidden_dim=100, action_dim=self.action_dim).to(device)

        self.lr = lr
        self.intrinsic_reward_rms = RMS(device=self.device)

        self.vail_optim = optim.Adam(
                                [
                                    {'params': self.discriminator.parameters()}
                                ],
                                lr=self.lr
                                )
        
        self.returns = None 
        self.ret_rms = RunningMeanStd(shape=())
        self.ob_rms = RunningMeanStd(shape=())

    def _bottleneck_loss(self, mus, sigmas, i_c=0.2, alpha=1e-8):
        """
        calculate the bottleneck loss for the given mus and sigmas
        :param mus: means of the gaussian distributions
        :param sigmas: stds of the gaussian distributions
        :param i_c: value of bottleneck
        :param alpha: small value for numerical stability
        :return: loss_value: scalar tensor
        """
        # add a small value to sigmas to avoid inf log
        kl_divergence = (0.5 * torch.sum((mus ** 2) + (sigmas ** 2)
                          - torch.log((sigmas ** 2) + alpha) - 1, dim=1))

        # calculate the bottleneck loss:
        bottleneck_loss = (torch.mean(kl_divergence) - i_c)

        # return the bottleneck_loss:
        return bottleneck_loss

    def compute_grad_pen(self,
                        expert_state,
                        expert_action,
                        policy_state,
                        policy_action,
                        lambda_=10):

       expert_data = torch.cat([expert_state, expert_action], dim=1)
       policy_data = torch.cat([policy_state, policy_action], dim=1)

       alpha = torch.rand_like(expert_data).to(expert_data.device)

       mixup_data = alpha * expert_data + (1 - alpha) * policy_data
       mixup_data.requires_grad = True

       disc,_,_ = self.discriminator(mixup_data)
       ones = torch.ones(disc.size()).to(disc.device)
       grad = autograd.grad(
           outputs=disc,
           inputs=mixup_data,
           grad_outputs=ones,
           create_graph=True,
           retain_graph=True,
           only_inputs=True)[0]

       grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
       return grad_pen
    
    def feed_forward_generator(self,
                               b_obs, 
                               b_actions,
                               num_minibatches,
                               minibatch_size=None):

        batch_size = b_obs.shape[1]
        if minibatch_size is None:
            minibatch_size = batch_size // num_minibatches
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            minibatch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = b_obs.view(-1, b_obs.shape[-1])[indices]
            actions_batch = b_actions.view(-1, b_actions.shape[-1])[indices]

            yield obs_batch, actions_batch

    def update(self, expert_loader, num_minibatches, b_obs, b_actions, obsfilt=None):
        policy_data_generator = self.feed_forward_generator(b_obs, b_actions, num_minibatches, 
                                                            expert_loader.batch_size)


        loss = 0
        expert_loss_sum = 0.0
        policy_loss_sum = 0.0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            policy_state, policy_action = policy_batch[0], policy_batch[1]
            policy_d, policy_mus, policy_sigmas = self.discriminator(
                torch.cat([policy_state, policy_action], dim=1))

            expert_state, expert_action, _ = expert_batch
            if obsfilt is not None:
                expert_state = obsfilt(expert_state.numpy(), update=False)
            
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_d, expert_mus, expert_sigmas = self.discriminator(
                torch.cat([expert_state, expert_action], dim=1))

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits( 
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))
            
            
            # calculate the bottleneck_loss:
            bottle_neck_loss = self._bottleneck_loss(
                            torch.cat((expert_mus, policy_mus), dim=0),
                                        torch.cat((expert_sigmas, policy_sigmas), dim=0), self.i_c)

            vail_loss = expert_loss + policy_loss + bottle_neck_loss
            grad_pen = self.compute_grad_pen(expert_state, expert_action,
                                            policy_state, policy_action)

            loss += (vail_loss + grad_pen).item()
            # loss += vail_loss.item()
            expert_loss_sum+=expert_loss.item()
            policy_loss_sum+=policy_loss.item()
            n += 1

            self.vail_optim.zero_grad()
            (vail_loss + grad_pen).backward()
            # vail_loss.backward()
            self.vail_optim.step()
        return loss / n , expert_loss_sum /n, policy_loss_sum/n, expert_sigmas

    def calculate_intrinsic_reward(self, state, action, 
                       use_original_reward=True, alpha=1e-8, reward_type='log1-d'):
        with torch.no_grad():
            d, mu, sigma = self.discriminator(torch.cat([state, action], dim=1))
            s = torch.sigmoid(d)  
            # solution from here: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/issues/204
            
            if reward_type == 'logd':
                reward =  torch.log(s + alpha)
            if reward_type == 'log1-d':
                reward =  - torch.log(1 - s + alpha)
            if self.returns is None:
                self.returns = reward.clone()

            if use_original_reward:
                return reward 
            else:
                return reward / np.sqrt(self.ret_rms.var[0] + alpha)

class ICM(object):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 hidden_dim=100,
                 device='cuda:0',
                 inverse_lr=3e-4,
                 forward_lr=3e-4,
                 ):

        self.device = device

        self.inverse_model = InverseModel(input_dim=obs_dim, action_dim=action_dim, \
                                     hidden_dim=hidden_dim).to(device)
        self.forward_dynamics_model = ForwardDynamicsModel(obs_dim=obs_dim, action_dim=action_dim, \
                                                      hidden_dim=hidden_dim).to(device)
        self.inverse_lr = inverse_lr
        self.forward_lr = forward_lr

        self.inverse_optim = optim.Adam(lr=self.inverse_lr, params=self.inverse_model.parameters())
        self.forward_optim = optim.Adam(lr=self.forward_lr, params=self.forward_dynamics_model.parameters())

    def get_inverse_dynamics_loss(self):
        criterionID = nn.MSELoss()
        return criterionID

    def get_forward_dynamics_loss(self):
        criterionFD = nn.MSELoss()
        return criterionFD

    def fit_batch(self, state, action, next_state, train=True):
        pred_action = self.inverse_model(state, next_state)
        criterionID = self.get_inverse_dynamics_loss()
        inverse_loss = criterionID(pred_action, action)
        if train:
            self.inverse_optim.zero_grad()
            inverse_loss.backward(retain_graph=True)
            self.inverse_optim.step()

        # Predict the next state from the current state and the action
        pred_next_state = self.forward_dynamics_model(state, action)
        criterionFD = self.get_forward_dynamics_loss()
        forward_loss = criterionFD(pred_next_state, next_state)
        if train:
            self.forward_optim.zero_grad()
            forward_loss.backward(retain_graph=True)
            self.forward_optim.step()

        return inverse_loss, forward_loss, pred_action

    # Calculation of the curiosity reward
    def calculate_intrinsic_reward(self, state, action, next_state):
        with torch.no_grad():
            if len(action.shape)>1:
                action = action.squeeze(1)

            pred_next_state = self.forward_dynamics_model(state, action)
            processed_next_state = process(next_state, normalize=True, range=(-1, 1))
            processed_pred_next_state = process(pred_next_state, normalize=True, range=(-1, 1))
            reward = F.mse_loss(processed_pred_next_state, processed_next_state, reduction='none')
            reward = torch.mean(reward, (0, 1, 3))

        return reward 

# measure conditioned ICM
class mCondICM(object):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 measure_dim,
                 hidden_dim=100,
                 device='cuda:0',
                 inverse_lr=3e-4,
                 forward_lr=3e-4,
                 ):

        self.device = device

        self.inverse_model = InverseModel(input_dim=obs_dim+measure_dim, action_dim=action_dim, \
                                     hidden_dim=hidden_dim).to(device)
        self.forward_dynamics_model = ForwardDynamicsModel(obs_dim=obs_dim+measure_dim, action_dim=action_dim, \
                                                      hidden_dim=hidden_dim).to(device)
        self.inverse_lr = inverse_lr
        self.forward_lr = forward_lr

        self.inverse_optim = optim.Adam(lr=self.inverse_lr, params=self.inverse_model.parameters())
        self.forward_optim = optim.Adam(lr=self.forward_lr, params=self.forward_dynamics_model.parameters())

    def get_inverse_dynamics_loss(self):
        criterionID = nn.MSELoss()
        return criterionID

    def get_forward_dynamics_loss(self):
        criterionFD = nn.MSELoss()
        return criterionFD

    def fit_batch(self, state, action, next_state, measure, next_measure, train=True):
        state_measure = torch.cat([state, measure], dim=-1)
        next_state_measure = torch.cat([next_state, next_measure], dim=-1)
        pred_action = self.inverse_model(state_measure, next_state_measure)
        criterionID = self.get_inverse_dynamics_loss()
        inverse_loss = criterionID(pred_action, action)
        if train:
            self.inverse_optim.zero_grad()
            inverse_loss.backward(retain_graph=True)
            self.inverse_optim.step()

        # Predict the next state from the current state and the action
        pred_next_state_measure = self.forward_dynamics_model(state_measure, action)
        criterionFD = self.get_forward_dynamics_loss()
        forward_loss = criterionFD(pred_next_state_measure, next_state_measure)
        if train:
            self.forward_optim.zero_grad()
            forward_loss.backward(retain_graph=True)
            self.forward_optim.step()

        return inverse_loss, forward_loss, pred_action

    # Calculation of the curiosity reward
    def calculate_intrinsic_reward(self, state, action, next_state, measure, next_measure):
        with torch.no_grad():
            if len(action.shape)>1:
                action = action.squeeze(1)
            state_measure = torch.cat([state, measure], dim=-1)
            next_state_measure = torch.cat([next_state, next_measure], dim=-1)
            pred_next_state_measure = self.forward_dynamics_model(state_measure, action)
            processed_next_state_measure = process(next_state_measure, normalize=True, range=(-1, 1))
            processed_pred_next_state_measure = process(pred_next_state_measure, normalize=True, range=(-1, 1))
            reward = F.mse_loss(processed_pred_next_state_measure, processed_next_state_measure, reduction='none')
            reward = torch.mean(reward, (0, 1, 3))

        return reward 

# measure regularized ICM, with measure_entropy bonus and measure_error bonus
class mRegICM(object):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 measure_dim,
                 reg_loss_fn='MSE',
                 bonus_type='measure_error',
                 hidden_dim=100,
                 device='cuda:0',
                 inverse_lr=3e-4,
                 forward_lr=3e-4,
                 ):

        self.device = device

        self.inverse_model = InverseModel(input_dim=obs_dim, action_dim=action_dim, \
                                     hidden_dim=hidden_dim).to(device)
        self.forward_dynamics_model = ForwardDynamicsModel(obs_dim=obs_dim, action_dim=action_dim, \
                                                      hidden_dim=hidden_dim).to(device)
        self.measure_predict_model = ForwardDynamicsModel(obs_dim=obs_dim, action_dim=action_dim, \
                                                      hidden_dim=hidden_dim, output_dim=measure_dim).to(device)
        self.inverse_lr = inverse_lr
        self.forward_lr = forward_lr

        self.reg_loss_fn = reg_loss_fn
        self.bonus_type = bonus_type
        if self.bonus_type == 'measure_entropy':
            rms = RMS(self.device)
            knn_rms = False
            knn_k = 500
            knn_avg = True
            knn_clip = 0.0
            self.pbe = PBE(rms, knn_clip, knn_k, knn_avg, knn_rms, self.device)

        if 'fitness_cond_measure_entropy' in self.bonus_type:
            knn_k = 500
            self.vcse = VCSE(knn_k)

        self.inverse_optim = optim.Adam(lr=self.inverse_lr, params=self.inverse_model.parameters())
        self.forward_optim = optim.Adam(
                                        [
                                            {'params': self.forward_dynamics_model.parameters()},
                                            {'params': self.measure_predict_model.parameters()}
                                        ],lr=self.forward_lr, 
                                        )

    def get_inverse_dynamics_loss(self):
        criterionID = nn.MSELoss()
        return criterionID

    def get_forward_dynamics_loss(self):
        criterionFD = nn.MSELoss()
        return criterionFD

    def fit_batch(self, state, action, next_state, measure, train=True):
        pred_action = self.inverse_model(state, next_state)
        criterionID = self.get_inverse_dynamics_loss()
        inverse_loss = criterionID(pred_action, action)
        if train:
            self.inverse_optim.zero_grad()
            inverse_loss.backward(retain_graph=True)
            self.inverse_optim.step()

        # Predict the next state from the current state and the action
        pred_next_state = self.forward_dynamics_model(state, action)
        pred_measure = self.measure_predict_model(state, action)
        criterionFD = self.get_forward_dynamics_loss()

        forward_loss = criterionFD(pred_next_state, next_state) 
        if self.reg_loss_fn == 'MSE':
            forward_loss += criterionFD(pred_measure, measure)
        if self.reg_loss_fn == 'NLL':
            forward_loss += NLL_loss(pred_measure, measure)
        if train:
            self.forward_optim.zero_grad()
            forward_loss.backward(retain_graph=True)
            self.forward_optim.step()

        return inverse_loss, forward_loss, pred_action

    # Calculation of the curiosity reward
    def calculate_intrinsic_reward(self, state, action, next_state, measure, value=None):
        with torch.no_grad():
            if len(action.shape)>1:
                action = action.squeeze(1)

            pred_next_state = self.forward_dynamics_model(state, action)
            
            processed_next_state = process(next_state, normalize=True, range=(-1, 1))
            processed_pred_next_state = process(pred_next_state, normalize=True, range=(-1, 1))
            icm_reward = F.mse_loss(processed_pred_next_state, processed_next_state, reduction='none')
            icm_reward = torch.mean(icm_reward, (0, 1, 3))

            if self.bonus_type == 'measure_error':
                pred_measure = self.measure_predict_model(state, action)
                bonus = F.mse_loss(pred_measure, measure,reduction='none').mean(dim=1)
            if self.bonus_type == 'measure_error_nll':
                bonus = NLL_loss(pred_measure, measure, reduction='none').mean(dim=1)
            if self.bonus_type == 'measure_entropy':
                bonus = self.pbe(measure).squeeze(1)
            if self.bonus_type == 'fitness_cond_measure_entropy' and value is not None:
                bonus = self.vcse(measure, value)[0].squeeze(1)
            if self.bonus_type == 'weighted_fitness_cond_measure_entropy' and value is not None:
                bonus_v, n_v,n_m, eps, state_norm, value_norm = self.vcse(measure, value)
                bonus_m = torch.digamma(n_m+1)/measure.shape[0] + torch.log(eps*2+0.00001)
                weight = 1/torch.abs(bonus_m)
                bonus = weight*bonus_v
                bonus = bonus.squeeze(1)
                

            reward = icm_reward + bonus
        return reward 

# measure conditioned and regularized ICM
class mCondRegICM(object):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 measure_dim,
                 reg_loss_fn='MSE',
                 bonus_type='measure_error',
                 hidden_dim=100,
                 device='cuda:0',
                 inverse_lr=3e-4,
                 forward_lr=3e-4,
                 ):

        self.device = device

        self.inverse_model = InverseModel(input_dim=obs_dim+measure_dim, action_dim=action_dim, \
                                     hidden_dim=hidden_dim).to(device)
        self.forward_dynamics_model = ForwardDynamicsModel(obs_dim=obs_dim+measure_dim, action_dim=action_dim, \
                                                      hidden_dim=hidden_dim).to(device)
        self.measure_predict_model = ForwardDynamicsModel(obs_dim=obs_dim, action_dim=action_dim, \
                                                      hidden_dim=hidden_dim, output_dim=measure_dim).to(device)
        self.inverse_lr = inverse_lr
        self.forward_lr = forward_lr

        self.reg_loss_fn = reg_loss_fn
        self.bonus_type = bonus_type
        if self.bonus_type == 'measure_entropy':
            rms = RMS(self.device)
            knn_rms = False
            knn_k = 500
            knn_avg = True
            knn_clip = 0.0
            self.pbe = PBE(rms, knn_clip, knn_k, knn_avg, knn_rms, self.device)

        if 'fitness_cond_measure_entropy' in self.bonus_type:
            knn_k = 500
            self.vcse = VCSE(knn_k)

        self.inverse_optim = optim.Adam(lr=self.inverse_lr, params=self.inverse_model.parameters())
        self.forward_optim = optim.Adam(
                                        [
                                            {'params': self.forward_dynamics_model.parameters()},
                                            {'params': self.measure_predict_model.parameters()}
                                        ],lr=self.forward_lr, 
                                        )

    def get_inverse_dynamics_loss(self):
        criterionID = nn.MSELoss()
        return criterionID

    def get_forward_dynamics_loss(self):
        criterionFD = nn.MSELoss()
        return criterionFD

    def fit_batch(self, state, action, next_state, measure, next_measure, train=True):
        state_measure = torch.cat([state, measure], dim=-1)
        next_state_measure = torch.cat([next_state, next_measure], dim=-1)
        pred_action = self.inverse_model(state_measure, next_state_measure)
        criterionID = self.get_inverse_dynamics_loss()
        inverse_loss = criterionID(pred_action, action)
        if train:
            self.inverse_optim.zero_grad()
            inverse_loss.backward(retain_graph=True)
            self.inverse_optim.step()

        # Predict the next state from the current state and the action
        pred_next_state_measure = self.forward_dynamics_model(state_measure, action)
        criterionFD = self.get_forward_dynamics_loss()
        pred_measure = self.measure_predict_model(state, action)
        forward_loss = criterionFD(pred_next_state_measure, next_state_measure) 

        if self.reg_loss_fn == 'MSE':
            forward_loss += criterionFD(pred_next_state_measure, next_state_measure)
        if self.reg_loss_fn == 'NLL':
            forward_loss += NLL_loss(pred_next_state_measure, next_state_measure)

        if train:
            self.forward_optim.zero_grad()
            forward_loss.backward(retain_graph=True)
            self.forward_optim.step()

        return inverse_loss, forward_loss, pred_action

    # Calculation of the curiosity reward
    def calculate_intrinsic_reward(self, state, action, next_state, measure, next_measure, value=None):
        with torch.no_grad():
            if len(action.shape)>1:
                action = action.squeeze(1)

            state_measure = torch.cat([state, measure], dim=-1)
            next_state_measure = torch.cat([next_state, next_measure], dim=-1)
            pred_next_state_measure = self.forward_dynamics_model(state_measure, action)
            processed_next_state_measure = process(next_state_measure, normalize=True, range=(-1, 1))
            processed_pred_next_state_measure = process(pred_next_state_measure, normalize=True, range=(-1, 1))
            icm_reward = F.mse_loss(processed_pred_next_state_measure, processed_next_state_measure, reduction='none')
            icm_reward = torch.mean(icm_reward, (0, 1, 3))

            if self.bonus_type == 'measure_error':
                pred_measure = self.measure_predict_model(state, action)
                bonus = F.mse_loss(pred_measure, measure,reduction='none').mean(dim=1)
            if self.bonus_type == 'measure_error_nll':
                bonus = NLL_loss(pred_measure, measure, reduction='none').mean(dim=1)
            if self.bonus_type == 'measure_entropy':
                bonus = self.pbe(measure).squeeze(1)
            if self.bonus_type == 'fitness_cond_measure_entropy' and value is not None:
                bonus = self.vcse(measure, value)[0].squeeze(1)
            if self.bonus_type == 'weighted_fitness_cond_measure_entropy' and value is not None:
                bonus_v, n_v,n_m, eps, state_norm, value_norm = self.vcse(measure, value)
                bonus_m = torch.digamma(n_m+1)/measure.shape[0] + torch.log(eps*2+0.00001)
                weight = 1/torch.abs(bonus_m)
                bonus = weight*bonus_v
                bonus = bonus.squeeze(1)
                
            reward = icm_reward + bonus
        return reward 


class PBE(object):
    """particle-based entropy based on knn normalized by running mean """
    def __init__(self, rms, knn_clip, knn_k, knn_avg, knn_rms, device):
        self.rms = rms
        self.knn_rms = knn_rms
        self.knn_k = knn_k
        self.knn_avg = knn_avg
        self.knn_clip = knn_clip
        self.device = device

    def __call__(self, rep):
        source = target = rep
        b1, b2 = source.size(0), target.size(0)
        # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
        sim_matrix = torch.norm(source[:, None, :].view(b1, 1, -1) -
                                target[None, :, :].view(1, b2, -1),
                                dim=-1,
                                p=2)
        # avoid index out of range
        self.knn_k = min(self.knn_k, sim_matrix.shape[0])
        reward, _ = sim_matrix.topk(self.knn_k,
                                    dim=1,
                                    largest=False,
                                    sorted=True)  # (b1, k)
        # reward, _ = sim_matrix.topk(self.knn_k,dim=1,largest=False,sorted=True)  # (b1, k)
        if not self.knn_avg:  # only keep k-th nearest neighbor
            reward = reward[:, -1]
            reward = reward.reshape(-1, 1)  # (b1, 1)
            reward /= self.rms(reward)[0] if self.knn_rms else 1.0
            reward = torch.maximum(
                reward - self.knn_clip,
                torch.zeros_like(reward).to(self.device)
            ) if self.knn_clip >= 0.0 else reward  # (b1, 1)
        else:  # average over all k nearest neighbors
            reward = reward.reshape(-1, 1)  # (b1 * k, 1)
            reward /= self.rms(reward)[0] if self.knn_rms else 1.0
            reward = torch.maximum(
                reward - self.knn_clip,
                torch.zeros_like(reward).to(
                    self.device)) if self.knn_clip >= 0.0 else reward
            reward = reward.reshape((b1, self.knn_k))  # (b1, k)
            reward = reward.mean(dim=1, keepdim=True)  # (b1, 1)
        reward = torch.log(reward + 1.0)
        return reward


### Accelerating Reinforcement Learning with Value-Conditional State Entropy Exploration
# https://sites.google.com/view/rl-vcse
class VCSE(object):
    def __init__(self, knn_k):
        self.knn_k = knn_k

    def __call__(self, state,value):
        #value => [b1 , 1]
        #state => [b1 , c]
        #z => [b1, c+1]
        # [b1] => [b1,b1]
        ds = state.size(1)
        source = target = state
        b1, b2 = source.size(0), target.size(0)
        # (b1, 1, c+1) - (1, b2, c+1) -> (b1, 1, c+1) - (1, b2, c+1) -> (b1, b2, c+1) -> (b1, b2)
        sim_matrix_s = torch.norm(source[:, None, :].view(b1, 1, -1) -
                                target[None, :, :].view(1, b2, -1),
                                dim=-1,
                                p=2)

        if len(value.shape) == 1:
            value = value.unsqueeze(1)
        
        source = target = value
        # (b1, 1, 1) - (1, b2, 1) -> (b1, 1, 1) - (1, b2, 1) -> (b1, b2, 1) -> (b1, b2)
        sim_matrix_v = torch.norm(source[:, None, :].view(b1, 1, -1) -
                                target[None, :, :].view(1, b2, -1),
                                dim=-1,
                                p=2)
        
        sim_matrix = torch.max(torch.cat((sim_matrix_s.unsqueeze(-1),sim_matrix_v.unsqueeze(-1)),dim=-1),dim=-1)[0]
        eps, index = sim_matrix.topk(self.knn_k,
                                    dim=1,
                                    largest=False,
                                    sorted=True)  # (b1, k)
        
        state_norm, index = sim_matrix_s.topk(self.knn_k,
                                    dim=1,
                                    largest=False,
                                    sorted=True)  # (b1, k)
        
        value_norm, index = sim_matrix_v.topk(self.knn_k,
                                    dim=1,
                                    largest=False,
                                    sorted=True)  # (b1, k)
        
        eps = eps[:, -1] #k-th nearest distance
        eps = eps.reshape(-1, 1) # (b1, 1)
        
        state_norm = state_norm[:, -1] #k-th nearest distance
        state_norm = state_norm.reshape(-1, 1) # (b1, 1)

        value_norm = value_norm[:, -1] #k-th nearest distance
        value_norm = value_norm.reshape(-1, 1) # (b1, 1)
        
        sim_matrix_v = sim_matrix_v < eps
        n_v = torch.sum(sim_matrix_v,dim=1,keepdim = True) # (b1,1)
        
        sim_matrix_s = sim_matrix_s < eps
        n_s = torch.sum(sim_matrix_s,dim=1,keepdim = True) # (b1,1)        

        reward = torch.digamma(n_v+1) / ds + torch.log(eps * 2 + 0.00001)
        return reward, n_v,n_s, eps, state_norm, value_norm  

class Encoder(nn.Module):

    def __init__(self, input_dim, action_dim,
                 hidden_dim=100):

        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = action_dim
        self.hidden = hidden_dim

        # Inverse Model architecture
        self.linear_1 = nn.Linear(in_features=self.input_dim*2, out_features=self.hidden)
        self.linear_2 = nn.Linear(in_features=self.hidden, out_features=self.hidden)
        
        self.tanh_1 = nn.Tanh()
        self.tanh_2 = nn.Tanh()

        self.mu = nn.Linear(hidden_dim, action_dim)
        self.logvar = nn.Linear(hidden_dim, action_dim)

        # Initialize the weights using xavier initialization
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.xavier_uniform_(self.mu.weight)
        nn.init.xavier_uniform_(self.logvar.weight)

    def reparameterize(self, mu, logvar, device, training=True):
        # Reparameterization trick as shown in the auto encoding variational bayes paper
        if training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_()).to(device)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, state, next_state):

        # Concatenate the state and the next state
        input = torch.cat([state, next_state], dim=-1)
        x = self.linear_1(input)
        x = self.tanh_1(x)
        x = self.linear_2(x)
        x = self.tanh_2(x)

        mu = self.mu(x)
        logvar = self.logvar(x)
        
        z = self.reparameterize(mu, logvar, state.device)
        return z, mu, logvar


class GIRIL(object):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 hidden_dim=100,
                 device='cuda:0',
                 lr=3e-4,
                 ):

        self.device = device
        self.encoder = Encoder(input_dim=obs_dim, action_dim=action_dim,
                                     hidden_dim=hidden_dim).to(device)
        self.forward_dynamics_model = ForwardDynamicsModel(obs_dim=obs_dim, action_dim=action_dim,
                                                      hidden_dim=hidden_dim).to(device)
        self.lr = lr
        

        self.optim = optim.Adam(
                                [
                                    {'params': self.encoder.parameters()},
                                    {'params': self.forward_dynamics_model.parameters()}
                                ],
                                lr=self.lr
                                )
    
    def get_vae_loss(self, recon_x, x, mean, log_var):
        RECON = F.mse_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    
        return RECON, KLD 

    def fit_batch(self, state, action, next_state, 
                  lambda_action=1.0, kld_loss_beta=1.0, train=True):
        
        z, mu, logvar = self.encoder(state, next_state)
         
        reconstructed_next_state = self.forward_dynamics_model(state, z)

        criterionAction = nn.MSELoss()
        action_loss = criterionAction(z, action)

        recon_loss, kld_loss = self.get_vae_loss(reconstructed_next_state, 
                                                 next_state, mu, logvar)
        vae_loss = recon_loss + kld_loss_beta * kld_loss + lambda_action * action_loss 

        if train:
            self.optim.zero_grad()
            vae_loss.backward(retain_graph=True)
            self.optim.step()

        # return inverse_loss, forward_loss, pred_action
        return vae_loss, recon_loss, kld_loss_beta*kld_loss, \
            lambda_action*action_loss, z

    # Calculation of the curiosity reward
    def calculate_intrinsic_reward(self, state, action, next_state):
        with torch.no_grad():
            if len(action.shape)>1:
                action = action.squeeze(1)

            pred_next_state = self.forward_dynamics_model(state, action)
            processed_next_state = process(next_state, normalize=True, range=(-1, 1))
            processed_pred_next_state = process(pred_next_state, normalize=True, range=(-1, 1))
            reward = F.mse_loss(processed_pred_next_state, processed_next_state, reduction='none')
            reward = torch.mean(reward, (0, 1, 3))
        return reward

class mCondGIRIL(object):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 measure_dim,
                 hidden_dim=100,
                 device='cuda:0',
                 lr=3e-4,
                 ):

        self.device = device
        self.encoder = Encoder(input_dim=obs_dim+measure_dim, action_dim=action_dim,
                                     hidden_dim=hidden_dim).to(device)
        self.forward_dynamics_model = ForwardDynamicsModel(obs_dim=obs_dim+measure_dim, action_dim=action_dim,
                                                      hidden_dim=hidden_dim).to(device)
        self.lr = lr
        

        self.optim = optim.Adam(
                                [
                                    {'params': self.encoder.parameters()},
                                    {'params': self.forward_dynamics_model.parameters()}
                                ],
                                lr=self.lr
                                )
    
    def get_vae_loss(self, recon_x, x, mean, log_var):
        RECON = F.mse_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    
        return RECON, KLD 

    def fit_batch(self, state, action, next_state, measure, next_measure,
                  lambda_action=1.0, kld_loss_beta=1.0, train=True):
        
        state_measure = torch.cat([state, measure], dim=-1)
        next_state_measure = torch.cat([next_state, next_measure], dim=-1)
        z, mu, logvar = self.encoder(state_measure, next_state_measure)
         
        reconstructed_next_state_measure = self.forward_dynamics_model(state_measure, z)

        criterionAction = nn.MSELoss()
        action_loss = criterionAction(z, action)

        recon_loss, kld_loss = self.get_vae_loss(reconstructed_next_state_measure, 
                                                 next_state_measure, mu, logvar)
        vae_loss = recon_loss + kld_loss_beta * kld_loss + lambda_action * action_loss 

        if train:
            self.optim.zero_grad()
            vae_loss.backward(retain_graph=True)
            self.optim.step()

        # return inverse_loss, forward_loss, pred_action
        return vae_loss, recon_loss, kld_loss_beta*kld_loss, \
            lambda_action*action_loss, z

    # Calculation of the curiosity reward
    def calculate_intrinsic_reward(self, state, action, next_state, measure, next_measure):
        with torch.no_grad():
            if len(action.shape)>1:
                action = action.squeeze(1)

            state_measure = torch.cat([state, measure], dim=-1)
            next_state_measure = torch.cat([next_state, next_measure], dim=-1)
            pred_next_state_measure = self.forward_dynamics_model(state_measure, action)
            processed_next_state_measure = process(next_state_measure, normalize=True, range=(-1, 1))
            processed_pred_next_state_measure = process(pred_next_state_measure, normalize=True, range=(-1, 1))
            reward = F.mse_loss(processed_pred_next_state_measure, processed_next_state_measure, reduction='none')
            reward = torch.mean(reward, (0, 1, 3))
        return reward


class mRegGIRIL(object):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 measure_dim,
                 reg_loss_fn='MSE',
                 bonus_type='measure_error',
                 hidden_dim=100,
                 device='cuda:0',
                 lr=3e-4,
                 ):

        self.device = device
        self.encoder = Encoder(input_dim=obs_dim, action_dim=action_dim,
                                     hidden_dim=hidden_dim).to(device)
        self.forward_dynamics_model = ForwardDynamicsModel(obs_dim=obs_dim, action_dim=action_dim,
                                                      hidden_dim=hidden_dim).to(device)
        self.measure_predict_model = ForwardDynamicsModel(obs_dim=obs_dim, action_dim=action_dim, \
                                                      hidden_dim=hidden_dim, output_dim=measure_dim).to(device)
        self.lr = lr

        self.reg_loss_fn = reg_loss_fn
        self.bonus_type = bonus_type
        if self.bonus_type == 'measure_entropy':
            rms = RMS(self.device)
            knn_rms = False
            knn_k = 500
            knn_avg = True
            knn_clip = 0.0
            self.pbe = PBE(rms, knn_clip, knn_k, knn_avg, knn_rms, self.device)

        if 'fitness_cond_measure_entropy' in self.bonus_type:
            knn_k = 500
            self.vcse = VCSE(knn_k)
        

        self.optim = optim.Adam(
                                [
                                    {'params': self.encoder.parameters()},
                                    {'params': self.forward_dynamics_model.parameters()}
                                ],
                                lr=self.lr
                                )
    
    def get_vae_loss(self, recon_x, x, mean, log_var, pred_measure, measure):
        RECON = F.mse_loss(recon_x, x)
        if self.reg_loss_fn == 'MSE':
            RECON += F.mse_loss(pred_measure, measure)
        if self.reg_loss_fn == 'NLL':
            RECON += NLL_loss(pred_measure, measure)
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    
        return RECON, KLD 

    def fit_batch(self, state, action, next_state, measure,
                  lambda_action=1.0, kld_loss_beta=1.0, train=True):
        
        z, mu, logvar = self.encoder(state, next_state)
         
        reconstructed_next_state = self.forward_dynamics_model(state, z)
        pred_measure = self.measure_predict_model(state, z)
        
        criterionAction = nn.MSELoss()
        action_loss = criterionAction(z, action)

        recon_loss, kld_loss = self.get_vae_loss(reconstructed_next_state, 
                                                 next_state, mu, logvar,
                                                 pred_measure, measure)
        vae_loss = recon_loss + kld_loss_beta * kld_loss + lambda_action * action_loss 

        if train:
            self.optim.zero_grad()
            vae_loss.backward(retain_graph=True)
            self.optim.step()

        # return inverse_loss, forward_loss, pred_action
        return vae_loss, recon_loss, kld_loss_beta*kld_loss, \
            lambda_action*action_loss, z

    # Calculation of the curiosity reward
    def calculate_intrinsic_reward(self, state, action, next_state, measure, value=None):
        with torch.no_grad():
            if len(action.shape)>1:
                action = action.squeeze(1)

            pred_next_state = self.forward_dynamics_model(state, action)
            processed_next_state = process(next_state, normalize=True, range=(-1, 1))
            processed_pred_next_state = process(pred_next_state, normalize=True, range=(-1, 1))
            giril_reward = F.mse_loss(processed_pred_next_state, processed_next_state, reduction='none')
            giril_reward = torch.mean(giril_reward, (0, 1, 3))

            if self.bonus_type == 'measure_error':
                pred_measure = self.measure_predict_model(state, action)
                bonus = F.mse_loss(pred_measure, measure,reduction='none').mean(dim=1)
            if self.bonus_type == 'measure_error_nll':
                bonus = NLL_loss(pred_measure, measure, reduction='none').mean(dim=1)
            if self.bonus_type == 'measure_entropy':
                bonus = self.pbe(measure).squeeze(1)
            if self.bonus_type == 'fitness_cond_measure_entropy' and value is not None:
                bonus = self.vcse(measure, value)[0].squeeze(1)
            if self.bonus_type == 'weighted_fitness_cond_measure_entropy' and value is not None:
                bonus_v, n_v,n_m, eps, state_norm, value_norm = self.vcse(measure, value)
                bonus_m = torch.digamma(n_m+1)/measure.shape[0] + torch.log(eps*2+0.00001)
                weight = 1/torch.abs(bonus_m)
                bonus = weight*bonus_v
                bonus = bonus.squeeze(1)
                
            reward = giril_reward + bonus
        return reward

class mCondRegGIRIL(object):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 measure_dim,
                 reg_loss_fn='MSE',
                 bonus_type='measure_error',
                 hidden_dim=100,
                 device='cuda:0',
                 lr=3e-4,
                 ):

        self.device = device
        self.encoder = Encoder(input_dim=obs_dim+measure_dim, action_dim=action_dim,
                                     hidden_dim=hidden_dim).to(device)
        self.forward_dynamics_model = ForwardDynamicsModel(obs_dim=obs_dim+measure_dim, action_dim=action_dim,
                                                      hidden_dim=hidden_dim).to(device)
        self.measure_predict_model = ForwardDynamicsModel(obs_dim=obs_dim, action_dim=action_dim, \
                                                      hidden_dim=hidden_dim, output_dim=measure_dim).to(device)
        self.lr = lr

        self.reg_loss_fn = reg_loss_fn
        self.bonus_type = bonus_type
        if self.bonus_type == 'measure_entropy':
            rms = RMS(self.device)
            knn_rms = False
            knn_k = 500
            knn_avg = True
            knn_clip = 0.0
            self.pbe = PBE(rms, knn_clip, knn_k, knn_avg, knn_rms, self.device)

        if 'fitness_cond_measure_entropy' in self.bonus_type:
            knn_k = 500
            self.vcse = VCSE(knn_k)
        

        self.optim = optim.Adam(
                                [
                                    {'params': self.encoder.parameters()},
                                    {'params': self.forward_dynamics_model.parameters()}
                                ],
                                lr=self.lr
                                )
    
    def get_vae_loss(self, recon_x, x, mean, log_var, pred_measure, measure):
        RECON = F.mse_loss(recon_x, x)
        if self.reg_loss_fn == 'MSE':
            RECON += F.mse_loss(pred_measure, measure)
        if self.reg_loss_fn == 'NLL':
            RECON += NLL_loss(pred_measure, measure)
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    
        return RECON, KLD 

    def fit_batch(self, state, action, next_state, measure, next_measure,
                  lambda_action=1.0, kld_loss_beta=1.0, train=True):
        
        state_measure = torch.cat([state, measure], dim=-1)
        next_state_measure = torch.cat([next_state, next_measure], dim=-1)
        z, mu, logvar = self.encoder(state_measure, next_state_measure)
         
        reconstructed_next_state_measure = self.forward_dynamics_model(state_measure, z)
        pred_measure = self.measure_predict_model(state, z)
        
        criterionAction = nn.MSELoss()
        action_loss = criterionAction(z, action)

        recon_loss, kld_loss = self.get_vae_loss(reconstructed_next_state_measure, 
                                                 next_state_measure, mu, logvar,
                                                 pred_measure, measure)
        vae_loss = recon_loss + kld_loss_beta * kld_loss + lambda_action * action_loss 

        if train:
            self.optim.zero_grad()
            vae_loss.backward(retain_graph=True)
            self.optim.step()

        # return inverse_loss, forward_loss, pred_action
        return vae_loss, recon_loss, kld_loss_beta*kld_loss, \
            lambda_action*action_loss, z

    # Calculation of the curiosity reward
    def calculate_intrinsic_reward(self, state, action, next_state, measure, next_measure, value=None):
        with torch.no_grad():
            if len(action.shape)>1:
                action = action.squeeze(1)

            state_measure = torch.cat([state, measure], dim=-1)
            next_state_measure = torch.cat([next_state, next_measure], dim=-1)
            pred_next_state_measure = self.forward_dynamics_model(state_measure, action)
            processed_next_state_measure = process(next_state_measure, normalize=True, range=(-1, 1))
            processed_pred_next_state_measure = process(pred_next_state_measure, normalize=True, range=(-1, 1))
            giril_reward = F.mse_loss(processed_pred_next_state_measure, processed_next_state_measure, reduction='none')
            giril_reward = torch.mean(giril_reward, (0, 1, 3))

            if self.bonus_type == 'measure_error':
                pred_measure = self.measure_predict_model(state, action)
                bonus = F.mse_loss(pred_measure, measure,reduction='none').mean(dim=1)
            if self.bonus_type == 'measure_error_nll':
                bonus = NLL_loss(pred_measure, measure, reduction='none').mean(dim=1)
            if self.bonus_type == 'measure_entropy':
                bonus = self.pbe(measure).squeeze(1)
            if self.bonus_type == 'fitness_cond_measure_entropy' and value is not None:
                bonus = self.vcse(measure, value)[0].squeeze(1)
            if self.bonus_type == 'weighted_fitness_cond_measure_entropy' and value is not None:
                bonus_v, n_v,n_m, eps, state_norm, value_norm = self.vcse(measure, value)
                bonus_m = torch.digamma(n_m+1)/measure.shape[0] + torch.log(eps*2+0.00001)
                weight = 1/torch.abs(bonus_m)
                bonus = weight*bonus_v
                bonus = bonus.squeeze(1)
                
            reward = giril_reward + bonus
        return reward

    
def parse_args():
    parser = argparse.ArgumentParser()
    # PPO params
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='ant')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--intrinsic_module', type=str, default='icm')
    parser.add_argument('--demo_dir', type=str, default='trajs_random_elite/100episodes/')
    parser.add_argument('--num_demo', type=int, default=100)
    parser.add_argument('--reward_save_dir', type=str, default='reward_100_random_elite/')
    parser.add_argument('--auxiliary_loss_fn', type=str, default='MSE', choices=['MSE', 'NLL'],
                        help='auxiliary loss function predicting measures.')
    parser.add_argument('--bonus_type', type=str, default='measure_error', 
                        help='bonus type for m_reg methods',
                        choices=['measure_error', 'measure_entropy'])
    parser.add_argument('--intrinsic_epoch', type=int, default=1000)
    parser.add_argument('--intrinsic_save_interval', type=int, default=50)
    parser.add_argument('--batch_size', default=32, type=int, help='batch size for loading data')
    

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
    # parser.add_argument('--grid_size', type=int, required=True, help='Number of cells per archive dimension')
    # parser.add_argument("--num_dims", type=int, required=True, help="Dimensionality of measures")
    # parser.add_argument("--popsize", type=int, required=True,
                        # help="Branching factor for each step of MEGA i.e. the number of branching solutions from the current solution point")
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


# measure conditioned VAIL
class mCondVAIL(object):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 measure_dim, 
                 bonus_type='single_step_archive_bonus',
                 i_c=0.5,
                 device='cuda:0',
                 lr=3e-4,
                 wo_a = False):
                 


        self.device = device
        self.i_c = i_c
        self.action_dim = action_dim
        
        self.discriminator = VAILdiscriminator(input_dim=obs_dim+measure_dim,  \
                                     hidden_dim=100, action_dim=self.action_dim).to(device)


        self.lr = lr
        self.intrinsic_reward_rms = RMS(device=self.device)

        self.vail_optim = optim.Adam(
                                [
                                    {'params': self.discriminator.parameters()}
                                ],
                                lr=self.lr
                                )
        
        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())
        self.ob_rms = RunningMeanStd(shape=())
        self.bonus_type = bonus_type
        self.env_info = {'obs_dim': obs_dim, 'action_dim': action_dim, 'measure_dim': measure_dim}
        self.single_step_archive = torch.ones([2]*measure_dim).to(self.device)#shape: 2*2*2*2...(measure_dim)

    def update_single_step_archive(self, single_step_measure):
        '''
        single_step_measure: tensor of shape (batch_size, measure_dim)
        e.g. [[1,1,0,0],[0,0,1,0]]
        '''
        indices = single_step_measure.t().long()  # Transpose to get the indices in the correct format
        values = torch.ones(single_step_measure.size(0)).to(self.device)  # Create a tensor of ones with size batch_size

        # Use scatter_add_ to accumulate values in the archive
        self.single_step_archive.index_put_(tuple(indices), values, accumulate=True)
    
    def calculate_single_step_bonus(self, single_step_measure):
        '''
        archive_distribution: tensor of shape (2,2,2,2,...,2) 2^measure_dim
        single_step_measure: tensor of shape (batch_size, measure_dim)
        e.g. [[1,1,0,0],[0,0,1,0]]
        
        return: tensor of shape (batch_size,)
        '''
        archive_distribution = self.single_step_archive / self.single_step_archive.sum()
        
        indices = list(single_step_measure.long().t()) # measure_dim * batch_size
        
        prob = archive_distribution[indices] # shape: (batch_size,)
        
        bonus = 1/(1 + prob) # is it good?
        return bonus

    def _bottleneck_loss(self, mus, sigmas, i_c=0.2, alpha=1e-8):
        """
        calculate the bottleneck loss for the given mus and sigmas
        :param mus: means of the gaussian distributions
        :param sigmas: stds of the gaussian distributions
        :param i_c: value of bottleneck
        :param alpha: small value for numerical stability
        :return: loss_value: scalar tensor
        """
        # add a small value to sigmas to avoid inf log
        kl_divergence = (0.5 * torch.sum((mus ** 2) + (sigmas ** 2)
                          - torch.log((sigmas ** 2) + alpha) - 1, dim=1))

        # calculate the bottleneck loss:
        bottleneck_loss = (torch.mean(kl_divergence) - i_c)

        # return the bottleneck_loss:
        return bottleneck_loss

    def compute_grad_pen(self,
                        expert_state,
                        expert_action,
                        policy_state,
                        policy_action,
                        lambda_=10):

       expert_data = torch.cat([expert_state, expert_action], dim=1)
       policy_data = torch.cat([policy_state, policy_action], dim=1)

       alpha = torch.rand_like(expert_data).to(expert_data.device)

       mixup_data = alpha * expert_data + (1 - alpha) * policy_data
       mixup_data.requires_grad = True

       disc, _, _ = self.discriminator(mixup_data)
       ones = torch.ones(disc.size()).to(disc.device)
       grad = autograd.grad(
           outputs=disc,
           inputs=mixup_data,
           grad_outputs=ones,
           create_graph=True,
           retain_graph=True,
           only_inputs=True)[0]

       grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
       return grad_pen

    def feed_forward_generator(self,
                               b_obs, 
                               b_actions,
                               b_measure,
                               num_minibatches,
                               minibatch_size=None):

        batch_size = b_obs.shape[1]
        if minibatch_size is None:
            minibatch_size = batch_size // num_minibatches
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            minibatch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = b_obs.view(-1, b_obs.shape[-1])[indices]
            actions_batch = b_actions.view(-1, b_actions.shape[-1])[indices]
            measure_batch = b_measure.view(-1, b_measure.shape[-1])[indices] 

            yield obs_batch, actions_batch, measure_batch


    def update(self, expert_loader, num_minibatches, b_obs, b_actions, b_measure, obsfilt=None):
        policy_data_generator = self.feed_forward_generator(b_obs, b_actions, b_measure, \
                                    num_minibatches, expert_loader.batch_size)

        loss = 0
        expert_loss_sum = 0.0
        policy_loss_sum = 0.0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            policy_state, policy_action, policy_measure = policy_batch[0], policy_batch[1], policy_batch[2]
            policy_d, policy_mus, policy_sigmas = self.discriminator(
                torch.cat([policy_state, policy_measure, policy_action], dim=1))

            expert_state, expert_action, expert_measure = expert_batch
            if obsfilt is not None:
                expert_state = obsfilt(expert_state.numpy(), update=False)
            
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_measure = expert_measure.to(self.device)
            expert_d, expert_mus, expert_sigmas = self.discriminator(
                torch.cat([expert_state, expert_measure, expert_action], dim=1))

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits( 
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))

            # calculate the bottleneck_loss:
            bottle_neck_loss = self._bottleneck_loss(
                            torch.cat((expert_mus, policy_mus), dim=0),
                                        torch.cat((expert_sigmas, policy_sigmas), dim=0), self.i_c)

            vail_loss = expert_loss + policy_loss + bottle_neck_loss
            grad_pen = self.compute_grad_pen(torch.cat([expert_state, expert_measure], dim=1), 
                                             expert_action,
                                             torch.cat([policy_state, policy_measure], dim=1), 
                                             policy_action)

            loss += (vail_loss + grad_pen).item()
            expert_loss_sum+=expert_loss.item()
            policy_loss_sum+=policy_loss.item()
            n += 1

            self.vail_optim.zero_grad()
            (vail_loss + grad_pen).backward()
            # vail_loss.backward()
            self.vail_optim.step()
        return loss / n , expert_loss_sum /n, policy_loss_sum/n, expert_sigmas

    def calculate_intrinsic_reward(self, state, action, measure,
                       use_original_reward=True, alpha=1e-8, reward_type='-log1-d'):
        with torch.no_grad():
            d, _, _ = self.discriminator(torch.cat([state, measure, action], dim=1))
            s = torch.sigmoid(d)  
            # solution from here: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/issues/204
            
            if reward_type == 'logd':
                gail_reward =  torch.log(s + alpha)
            if reward_type == '-log1-d':
                gail_reward =  - torch.log(1 - s + alpha)
            if reward_type == 'logd-log1-d':
                gail_reward =  torch.log(s + alpha) - torch.log(1 - s + alpha)

            if self.bonus_type == 'single_step_archive_bonus':
                bonus = self.calculate_single_step_bonus(measure)
                self.update_single_step_archive(measure)
                reward = gail_reward + bonus.unsqueeze(1) 
            else:
                reward = gail_reward

            if self.returns is None:
                self.returns = reward.clone()

            if use_original_reward:
                return reward 
            else:
                return reward / np.sqrt(self.ret_rms.var[0] + alpha)
if __name__ == '__main__':

    args = parse_args()

    device = torch.device("cuda:0")
    print('using device: %s' % device)
    
    args = parse_args()
    args.num_emitters = 1

    vec_env = make_vec_env_brax(args)
    args.obs_shape = vec_env.single_observation_space.shape
    args.action_shape = vec_env.single_action_space.shape
    obs_dim = args.obs_shape[0]
    action_dim = args.action_shape[0]
    print(obs_dim, action_dim)
    os.makedirs(args.reward_save_dir, exist_ok=True)
    intrinsic_log_file_name = f'{args.reward_save_dir}/reward_model_{args.intrinsic_module}_{args.env_name}.csv'
    reward_model_file_name = f'{args.reward_save_dir}/reward_model_{args.intrinsic_module}_{args.env_name}.pt'
    # if 'm_reg' in args.intrinsic_module:
    #     reward_model_file_name = f'{args.reward_save_dir}/reward_model_{args.intrinsic_module}_{args.reward_type}_{args.env_name}.pt'

    short_env_name = args.env_name.replace('-v2', '')
    measure_dim = 2
    if 'ant' in short_env_name:
        measure_dim = 4
    if 'hopper' in short_env_name:
        measure_dim = 1

    if args.intrinsic_module == 'icm':
        reward_model = ICM(
                           obs_dim, 
                            action_dim,
                            inverse_lr=3e-4,
                            forward_lr=3e-4)
        
        with open(intrinsic_log_file_name, 'w') as f:
            head = 'Epoch,Inverse_loss,Forward_loss,IntrinsicReward\n'
            f.write(head)
            
        dataset, dataloader = load_sa_data(args, return_next_state=True)
        for e in range(args.intrinsic_epoch):
            e_inverse_loss = 0
            e_forward_loss = 0
            e_reward = 0
            for i, (state, action, next_state, measure, next_measure) in enumerate(dataloader):
                state = state.to(device)
                action = action.to(device)
                next_state = next_state.to(device)
                
                inverse_loss, forward_loss, action_logit = reward_model.fit_batch(state, action, next_state)
                rewards = reward_model.calculate_intrinsic_reward(state, action, next_state)
                e_inverse_loss += inverse_loss.item()
                e_forward_loss += forward_loss.item()
                e_reward += torch.mean(rewards).item()
            
            e_inverse_loss /= (i+1)
            e_forward_loss /= (i+1)
            e_reward /= (i+1)
            if (e+1) % 50 == 0:
                print('Epoch:', e+1, '%s-%s Loss - Inverse loss: %s and Forward loss: %s,  Rewards: %s' \
                    % ( args.intrinsic_module, short_env_name, \
                        e_inverse_loss, e_forward_loss, e_reward))
            
                result_str = f'{e+1},{e_inverse_loss},{e_forward_loss},{e_reward}\n'
                with open(intrinsic_log_file_name, 'a') as f:
                    f.write(result_str)
            if (e+1) % args.intrinsic_save_interval == 0:
                print('Saving the pretrained %s epochs %s as %s' % (e+1, args.intrinsic_module, \
                                                                    reward_model_file_name))
                torch.save([reward_model.inverse_model, reward_model.forward_dynamics_model], reward_model_file_name)
             

    if args.intrinsic_module == 'm_cond_icm':
        reward_model = mCondICM(
                           obs_dim, 
                            action_dim,
                            measure_dim,
                            inverse_lr=3e-4,
                            forward_lr=3e-4)
        
        with open(intrinsic_log_file_name, 'w') as f:
            head = 'Epoch,Inverse_loss,Forward_loss,IntrinsicReward\n'
            f.write(head)
            
        dataset, dataloader = load_sa_data(args, return_next_state=True)
        for e in range(args.intrinsic_epoch):
            e_inverse_loss = 0
            e_forward_loss = 0
            e_reward = 0
            for i, (state, action, next_state, measure, next_measure) in enumerate(dataloader):
                state = state.to(device)
                action = action.to(device)
                next_state = next_state.to(device)
                measure = measure.to(device)
                next_measure = next_measure.to(device)
                inverse_loss, forward_loss, action_logit = \
                    reward_model.fit_batch(state, action, next_state, measure, next_measure)
                rewards = reward_model.calculate_intrinsic_reward(state, action, next_state, measure, next_measure)
                e_inverse_loss += inverse_loss.item()
                e_forward_loss += forward_loss.item()
                e_reward += torch.mean(rewards).item()
            
            e_inverse_loss /= (i+1)
            e_forward_loss /= (i+1)
            e_reward /= (i+1)
            if (e+1) % 50 == 0:
                print('Epoch:', e+1, '%s-%s Loss - Inverse loss: %s and Forward loss: %s,  Rewards: %s' \
                    % ( args.intrinsic_module, short_env_name, \
                        e_inverse_loss, e_forward_loss, e_reward))
            
                result_str = f'{e+1},{e_inverse_loss},{e_forward_loss},{e_reward}\n'
                with open(intrinsic_log_file_name, 'a') as f:
                    f.write(result_str)
            if (e+1) % args.intrinsic_save_interval == 0:
                print('Saving the pretrained %s epochs %s as %s' % (e+1, args.intrinsic_module, \
                                                                    reward_model_file_name))
                torch.save([reward_model.inverse_model, reward_model.forward_dynamics_model], reward_model_file_name)
             
    if args.intrinsic_module == 'm_reg_icm':
        reward_model = mRegICM(
                           obs_dim, 
                            action_dim,
                            measure_dim,
                            reg_loss_fn=args.auxiliary_loss_fn,
                            bonus_type=args.bonus_type,
                            inverse_lr=3e-4,
                            forward_lr=3e-4)
        
        with open(intrinsic_log_file_name, 'w') as f:
            head = 'Epoch,Inverse_loss,Forward_loss,IntrinsicReward\n'
            f.write(head)
            
        dataset, dataloader = load_sa_data(args, return_next_state=True)
        for e in range(args.intrinsic_epoch):
            e_inverse_loss = 0
            e_forward_loss = 0
            e_reward = 0
            for i, (state, action, next_state, measure, next_measure) in enumerate(dataloader):
                state = state.to(device)
                action = action.to(device)
                next_state = next_state.to(device)
                measure = measure.to(device)
                next_measure = next_measure.to(device)
                inverse_loss, forward_loss, action_logit = \
                    reward_model.fit_batch(state, action, next_state, measure)
                rewards = reward_model.calculate_intrinsic_reward(state, action, next_state, measure)
                e_inverse_loss += inverse_loss.item()
                e_forward_loss += forward_loss.item()
                e_reward += torch.mean(rewards).item()
            
            e_inverse_loss /= (i+1)
            e_forward_loss /= (i+1)
            e_reward /= (i+1)
            if (e+1) % 50 == 0:
                print('Epoch:', e+1, '%s-%s Loss - Inverse loss: %s and Forward loss: %s,  Rewards: %s' \
                    % ( args.intrinsic_module, short_env_name, \
                        e_inverse_loss, e_forward_loss, e_reward))
            
                result_str = f'{e+1},{e_inverse_loss},{e_forward_loss},{e_reward}\n'
                with open(intrinsic_log_file_name, 'a') as f:
                    f.write(result_str)
            if (e+1) % args.intrinsic_save_interval == 0:
                print('Saving the pretrained %s epochs %s as %s' % (e+1, args.intrinsic_module, \
                                                                    reward_model_file_name))
                torch.save([reward_model.inverse_model, reward_model.forward_dynamics_model], reward_model_file_name)
             
    if args.intrinsic_module == 'm_cond_reg_icm':
        reward_model = mCondRegICM(
                           obs_dim, 
                            action_dim,
                            measure_dim,
                            reg_loss_fn=args.auxiliary_loss_fn,
                            bonus_type=args.bonus_type,
                            inverse_lr=3e-4,
                            forward_lr=3e-4)
        
        with open(intrinsic_log_file_name, 'w') as f:
            head = 'Epoch,Inverse_loss,Forward_loss,IntrinsicReward\n'
            f.write(head)
            
        dataset, dataloader = load_sa_data(args, return_next_state=True)
        for e in range(args.intrinsic_epoch):
            e_inverse_loss = 0
            e_forward_loss = 0
            e_reward = 0
            for i, (state, action, next_state, measure, next_measure) in enumerate(dataloader):
                state = state.to(device)
                action = action.to(device)
                next_state = next_state.to(device)
                measure = measure.to(device)
                next_measure = next_measure.to(device)
                inverse_loss, forward_loss, action_logit = \
                    reward_model.fit_batch(state, action, next_state, measure, next_measure)
                rewards = reward_model.calculate_intrinsic_reward(state, action, next_state, measure, next_measure)
                e_inverse_loss += inverse_loss.item()
                e_forward_loss += forward_loss.item()
                e_reward += torch.mean(rewards).item()
            
            e_inverse_loss /= (i+1)
            e_forward_loss /= (i+1)
            e_reward /= (i+1)
            if (e+1) % 50 == 0:
                print('Epoch:', e+1, '%s-%s Loss - Inverse loss: %s and Forward loss: %s,  Rewards: %s' \
                    % ( args.intrinsic_module, short_env_name, \
                        e_inverse_loss, e_forward_loss, e_reward))
            
                result_str = f'{e+1},{e_inverse_loss},{e_forward_loss},{e_reward}\n'
                with open(intrinsic_log_file_name, 'a') as f:
                    f.write(result_str)
            if (e+1) % args.intrinsic_save_interval == 0:
                print('Saving the pretrained %s epochs %s as %s' % (e+1, args.intrinsic_module, \
                                                                    reward_model_file_name))
                torch.save([reward_model.inverse_model, reward_model.forward_dynamics_model], reward_model_file_name)
             
    if args.intrinsic_module == 'giril':
        reward_model = GIRIL(
                           obs_dim, 
                            action_dim,
                            lr=3e-4)
        
        with open(intrinsic_log_file_name, 'w') as f:
            head = 'Epoch,VAE_loss,Recon_loss,KLD_loss,Action_loss,IntrinsicReward\n'
            f.write(head)
            
        dataset, dataloader = load_sa_data(args, return_next_state=True)
        for e in range(args.intrinsic_epoch):
            e_vae_loss = 0
            e_recon_loss = 0
            e_kld_loss = 0
            e_action_loss = 0
            e_reward = 0
            for i, (state, action, next_state, measure, next_measure) in enumerate(dataloader):
                state = state.to(device)
                action = action.to(device)
                next_state = next_state.to(device)
                vae_loss, recon_loss, kld_loss, action_loss, action_logit = \
                    reward_model.fit_batch(state, action, next_state,
                                           lambda_action=0.01,
                                           kld_loss_beta=1.0
                                           )

                rewards = reward_model.calculate_intrinsic_reward(state, action, next_state)[0]
                e_vae_loss += vae_loss.item()
                e_recon_loss += recon_loss.item()
                e_kld_loss += kld_loss.item()
                e_action_loss += action_loss.item()
                e_reward += torch.mean(rewards).item()
            
            e_vae_loss /= (i+1)
            e_recon_loss /= (i+1)
            e_kld_loss /= (i+1)
            e_action_loss /= (i+1)
            e_reward /= (i+1)
            if (e+1) % 50 == 0:
                print('Epoch:', e+1, '%s-%s Loss - VAE loss: %s, Recon loss: %s, KLD loss: %s and Action loss: %s,  Rewards: %s' \
                    % ( args.intrinsic_module, short_env_name, \
                        e_vae_loss, e_recon_loss, e_kld_loss, e_action_loss, e_reward))
            
                result_str = f'{e+1},{e_vae_loss},{e_recon_loss},{e_kld_loss},{e_action_loss},{e_reward}\n'
                with open(intrinsic_log_file_name, 'a') as f:
                    f.write(result_str)
            if (e+1) % args.intrinsic_save_interval == 0:
                print('Saving the pretrained %s epochs %s as %s' % (e+1, args.intrinsic_module, \
                                                                    reward_model_file_name))
                torch.save([reward_model.encoder, reward_model.forward_dynamics_model], reward_model_file_name)             

    if args.intrinsic_module == 'm_cond_giril':
        reward_model = mCondGIRIL(
                           obs_dim, 
                            action_dim,
                            measure_dim,
                            lr=3e-4)
        
        with open(intrinsic_log_file_name, 'w') as f:
            head = 'Epoch,VAE_loss,Recon_loss,KLD_loss,Action_loss,IntrinsicReward\n'
            f.write(head)
            
        dataset, dataloader = load_sa_data(args, return_next_state=True)
        for e in range(args.intrinsic_epoch):
            e_vae_loss = 0
            e_recon_loss = 0
            e_kld_loss = 0
            e_action_loss = 0
            e_reward = 0
            for i, (state, action, next_state, measure, next_measure) in enumerate(dataloader):
                state = state.to(device)
                action = action.to(device)
                next_state = next_state.to(device)
                measure = measure.to(device)
                next_measure = next_measure.to(device)
                vae_loss, recon_loss, kld_loss, action_loss, action_logit = \
                    reward_model.fit_batch(state, action, next_state, measure, next_measure,
                                           lambda_action=0.01,
                                           kld_loss_beta=1.0
                                           )

                rewards = reward_model.calculate_intrinsic_reward(state, action, next_state, 
                                                                  measure, next_measure)[0]
                e_vae_loss += vae_loss.item()
                e_recon_loss += recon_loss.item()
                e_kld_loss += kld_loss.item()
                e_action_loss += action_loss.item()
                e_reward += torch.mean(rewards).item()
            
            e_vae_loss /= (i+1)
            e_recon_loss /= (i+1)
            e_kld_loss /= (i+1)
            e_action_loss /= (i+1)
            e_reward /= (i+1)
            if (e+1) % 50 == 0:
                print('Epoch:', e+1, '%s-%s Loss - VAE loss: %s, Recon loss: %s, KLD loss: %s and Action loss: %s,  Rewards: %s' \
                    % ( args.intrinsic_module, short_env_name, \
                        e_vae_loss, e_recon_loss, e_kld_loss, e_action_loss, e_reward))
            
                result_str = f'{e+1},{e_vae_loss},{e_recon_loss},{e_kld_loss},{e_action_loss},{e_reward}\n'
                with open(intrinsic_log_file_name, 'a') as f:
                    f.write(result_str)
            if (e+1) % args.intrinsic_save_interval == 0:
                print('Saving the pretrained %s epochs %s as %s' % (e+1, args.intrinsic_module, \
                                                                    reward_model_file_name))
                torch.save([reward_model.encoder, reward_model.forward_dynamics_model], reward_model_file_name)             


    if args.intrinsic_module == 'm_reg_giril':
        reward_model = mRegGIRIL(
                            obs_dim, 
                            action_dim,
                            measure_dim,
                            reg_loss_fn=args.auxiliary_loss_fn,
                            bonus_type=args.bonus_type,
                            lr=3e-4)
        
        with open(intrinsic_log_file_name, 'w') as f:
            head = 'Epoch,VAE_loss,Recon_loss,KLD_loss,Action_loss,IntrinsicReward\n'
            f.write(head)
            
        dataset, dataloader = load_sa_data(args, return_next_state=True)
        for e in range(args.intrinsic_epoch):
            e_vae_loss = 0
            e_recon_loss = 0
            e_kld_loss = 0
            e_action_loss = 0
            e_reward = 0
            for i, (state, action, next_state, measure, next_measure) in enumerate(dataloader):
                state = state.to(device)
                action = action.to(device)
                next_state = next_state.to(device)
                measure = measure.to(device)

                vae_loss, recon_loss, kld_loss, action_loss, action_logit = \
                    reward_model.fit_batch(state, action, next_state, measure, 
                                           lambda_action=0.01,
                                           kld_loss_beta=1.0
                                           )

                rewards = reward_model.calculate_intrinsic_reward(state, action, next_state, measure)[0]
                e_vae_loss += vae_loss.item()
                e_recon_loss += recon_loss.item()
                e_kld_loss += kld_loss.item()
                e_action_loss += action_loss.item()
                e_reward += torch.mean(rewards).item()
            
            e_vae_loss /= (i+1)
            e_recon_loss /= (i+1)
            e_kld_loss /= (i+1)
            e_action_loss /= (i+1)
            e_reward /= (i+1)
            if (e+1) % 50 == 0:
                print('Epoch:', e+1, '%s-%s Loss - VAE loss: %s, Recon loss: %s, KLD loss: %s and Action loss: %s,  Rewards: %s' \
                    % ( args.intrinsic_module, short_env_name, \
                        e_vae_loss, e_recon_loss, e_kld_loss, e_action_loss, e_reward))
            
                result_str = f'{e+1},{e_vae_loss},{e_recon_loss},{e_kld_loss},{e_action_loss},{e_reward}\n'
                with open(intrinsic_log_file_name, 'a') as f:
                    f.write(result_str)
            if (e+1) % args.intrinsic_save_interval == 0:
                print('Saving the pretrained %s epochs %s as %s' % (e+1, args.intrinsic_module, \
                                                                    reward_model_file_name))
                torch.save([reward_model.encoder, reward_model.forward_dynamics_model], reward_model_file_name)             


    if args.intrinsic_module == 'm_cond_reg_giril':
        reward_model = mCondRegGIRIL(
                            obs_dim, 
                            action_dim,
                            measure_dim,
                            reg_loss_fn=args.auxiliary_loss_fn,
                            bonus_type=args.bonus_type,
                            lr=3e-4)
        
        with open(intrinsic_log_file_name, 'w') as f:
            head = 'Epoch,VAE_loss,Recon_loss,KLD_loss,Action_loss,IntrinsicReward\n'
            f.write(head)
            
        dataset, dataloader = load_sa_data(args, return_next_state=True)
        for e in range(args.intrinsic_epoch):
            e_vae_loss = 0
            e_recon_loss = 0
            e_kld_loss = 0
            e_action_loss = 0
            e_reward = 0
            for i, (state, action, next_state, measure, next_measure) in enumerate(dataloader):
                state = state.to(device)
                action = action.to(device)
                next_state = next_state.to(device)
                measure = measure.to(device)
                next_measure = next_measure.to(device)

                vae_loss, recon_loss, kld_loss, action_loss, action_logit = \
                    reward_model.fit_batch(state, action, next_state, measure, next_measure,
                                           lambda_action=0.01,
                                           kld_loss_beta=1.0
                                           )

                rewards = reward_model.calculate_intrinsic_reward(state, action, next_state, 
                                                                  measure, next_measure)[0]
                e_vae_loss += vae_loss.item()
                e_recon_loss += recon_loss.item()
                e_kld_loss += kld_loss.item()
                e_action_loss += action_loss.item()
                e_reward += torch.mean(rewards).item()
            
            e_vae_loss /= (i+1)
            e_recon_loss /= (i+1)
            e_kld_loss /= (i+1)
            e_action_loss /= (i+1)
            e_reward /= (i+1)
            if (e+1) % 50 == 0:
                print('Epoch:', e+1, '%s-%s Loss - VAE loss: %s, Recon loss: %s, KLD loss: %s and Action loss: %s,  Rewards: %s' \
                    % ( args.intrinsic_module, short_env_name, \
                        e_vae_loss, e_recon_loss, e_kld_loss, e_action_loss, e_reward))
            
                result_str = f'{e+1},{e_vae_loss},{e_recon_loss},{e_kld_loss},{e_action_loss},{e_reward}\n'
                with open(intrinsic_log_file_name, 'a') as f:
                    f.write(result_str)
            if (e+1) % args.intrinsic_save_interval == 0:
                print('Saving the pretrained %s epochs %s as %s' % (e+1, args.intrinsic_module, \
                                                                    reward_model_file_name))
                torch.save([reward_model.encoder, reward_model.forward_dynamics_model], reward_model_file_name)             
