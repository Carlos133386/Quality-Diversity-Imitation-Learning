from __future__ import print_function
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
import pickle as pickle
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pdb

class ExpertDataset(torch.utils.data.TensorDataset):
    """

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        action_transform (callable, optional): A function/transform that takes in the
            action and transforms it.
    """
    def __init__(self, file_name, num_trajectories=10, train=True, train_test_split=1.0, \
        subsample_frequency=20, transform=None, action_transform=None, \
            return_next_state=False):

        traj_data = pickle.load(open(file_name, 'rb'))
        print('loaded demonstrations with returns:', [r.item() for r in traj_data['returns']])
        print('loaded demonstrations with lengths:', traj_data['lengths'])

        self.transform = transform
        self.action_transform = action_transform
        self.train = train  # training set or test set
        self.split = train_test_split
        self.return_next_state = return_next_state

        acc_eps_lengths = []
        length = 0
        for l in traj_data['lengths']:
            length += l
            acc_eps_lengths.append(length)
        
        idx = acc_eps_lengths[num_trajectories-1]

        start_idx = torch.randint(
            0, subsample_frequency, size=(1, )).long()

        self.trajectories = {}

        for k, v in traj_data.items():
            if k in ['states', 'measures']:
                state_data = v[:idx-1]
                next_state_data = v[1:idx]
                self.trajectories[k] = torch.from_numpy(state_data[start_idx::subsample_frequency]).float()
                self.trajectories[f'next_{k}'] = torch.from_numpy(next_state_data[start_idx::subsample_frequency]).float()
            elif k in ['actions', 'rewards']: 
                data = v[:idx-1]
                self.trajectories[k] = torch.from_numpy(data[::subsample_frequency]).float()
            else:
                data = traj_data[k][:num_trajectories]
                self.trajectories[k] = torch.from_numpy(data).float()

        self.length = self.trajectories['states'].shape[0]
        if self.train:
            self.train_states = self.trajectories['states'][:int(self.length*self.split)]
            self.train_actions = self.trajectories['actions'][:int(self.length*self.split)]
            self.train_next_states = self.trajectories['next_states'][:int(self.length*self.split)]
            self.train_measures = self.trajectories['measures'][:int(self.length*self.split)]
            self.train_next_measures = self.trajectories['next_measures'][:int(self.length*self.split)]
            print('Total training states: %s' %(self.train_states.shape[0]))
            
        else:
            self.test_states = self.trajectories['states'][int(self.length*self.split):]
            self.test_actions = self.trajectories['actions'][int(self.length*self.split):]
            self.test_next_states = self.trajectories['next_states'][int(self.length*self.split):]
            self.test_measures = self.trajectories['measures'][int(self.length*self.split):]
            self.test_next_measures = self.trajectories['next_measures'][int(self.length*self.split):]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (state, action) where action is index of the action class.
        """
        if self.train:
            state, action, next_state, measure, next_measure = \
                self.train_states[index], self.train_actions[index], self.train_next_states[index], \
                    self.train_measures[index], self.train_next_measures[index]
        else:
            state, action, next_state, measure, next_measure = \
                self.test_states[index], self.test_actions[index], self.test_next_states[index], \
                    self.test_measures[index], self.test_measures[index]

        if self.return_next_state:
            return state, action, next_state, measure, next_measure
        else:
            return state, action, measure

    def __len__(self):
        if self.train:
            return len(self.train_states)
        else:
            return len(self.test_states)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.action_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


if __name__ == "__main__":
    traj_root = 'trajs_good_and_diverse_elite_with_measures_top500/4episodes'
    env_name='humanoid'
    traj_file = f'{traj_root}/trajs_ppga_{env_name}.pt'
    print(f'Loading data: {traj_file}')
    dataset = ExpertDataset(file_name=traj_file,num_trajectories=1, train=True, train_test_split=1.0, return_next_state=True)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=False, num_workers=1, drop_last=True)
    pdb.set_trace()
    for state, action, next_state, measure, next_measure in dataloader:
        print(state.shape, '==')
        print(action.shape)
        print(next_state.shape)
        print(measure.shape)
        print(next_measure.shape)