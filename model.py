import torch
import torch.autograd as autograd
import torch.nn as nn
from torch import Tensor
import numpy as np
from torch.autograd import Variable
import math
from collections import namedtuple
import os
import datetime

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state', 'reward'))
OldCopy = namedtuple('OldCopy', ('log_density', 'action_mean', 'action_log_std', 'action_std'))

class args(object):
    env_name = 'Hopper-v2'
    seed = 1234
    num_episode = 500
    batch_size = 5000
    max_step_per_episode = 200
    gamma = 0.995
    lamda = 0.97
    l2_reg = 1e-3
    value_opt_max_iter = 25
    damping = 0.1
    max_kl = 1e-2
    cg_nsteps = 10
    log_num_episode = 1
    num_parallel_run = 5

class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)

        self.action_mean = nn.Linear(64, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

        self.saved_actions = []
        self.rewards = []
        self.final_value = 0

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    @staticmethod
    def _normal_log_density(x, mean, log_std, std):
        """
        returns probability distribution of normal distribution N(x; mean, std)
        :param x, mean, log_std, std: torch.Tensor
        :return: log probability density torch.Tensor
        """
        var = std.pow(2)
        log_density = - 0.5 * math.log(2 * math.pi) - log_std - (x - mean).pow(2) / (2 * var) 
        return log_density.sum(1)

    def _get_log_p_a_s(self, states, actions, return_net_output=False):
        """
        get log p(a|s) on data (states, actions)
        :param states, actions: torch.Tensor
        :return: log probability density torch.Tensor 
        """
        action_means, action_log_stds, action_stds = self.__call__(states)
        log_density = self._normal_log_density(actions, action_means, action_log_stds, action_stds)
        if return_net_output:
            return OldCopy(log_density=Variable(log_density), 
                action_mean=Variable(action_means), 
                action_log_std=Variable(action_log_stds), 
                action_std=Variable(action_stds))
        else:
            return log_density

    def set_old_loss(self, states, actions):
        self.old_copy = self._get_log_p_a_s(states, actions, return_net_output=True)

    def get_loss(self, states, actions, advantages):
        """
        get loss variable
        loss = dfrac{pi_theta (a|s)}{q(a|s)} Q(s, a)

        :param states: torch.Tensor
        :param actions: torch.Tensor
        :param advantages: torch.Tensor
        :return: the loss, torch.Variable
        """
        assert self.old_copy is not None
        log_prob = self._get_log_p_a_s(states, actions)
        # notice Variable(x) here means x is treated as fixed data 
        # and autograd is not applied to parameters that generated x.
        # in another word, pi_{theta_old}(a|s) is fixed and the gradient is taken w.r.t. new theta
        action_loss = - advantages * torch.exp(log_prob - self.old_copy.log_density)
        return action_loss.mean()

    def get_kl(self, states):
        """
        given old and new (mean, log_std, std) calculate KL divergence 
        pay attention 
            1. the distribution is a normal distribution on a continuous domain
            2. the KL divergence is a integration over (-inf, inf) 
                KL = integrate p0(x) log(p0(x) / p(x)) dx
        thus, KL can be calculated analytically
                KL = log_std - log_std0 + (var0 + (mean - mean0)^2) / (2 var) - 1/2
        ref: https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians

        :param states: torch.Tensor(#samples, #d_state)
        :return: KL torch.Tensor(1)
        """
        action_mean, action_log_std, action_std = self.__call__(states)
        kl = action_log_std - self.old_copy.action_log_std \
            + (self.old_copy.action_std.pow(2) + (self.old_copy.action_mean - action_mean).pow(2)) \
            / (2.0 * action_std.pow(2)) - 0.5
        return kl.sum(1).mean()

    def kl_hessian_times_vector(self, states, v):
        """
        return the product of KL's hessian and an arbitrary vector in O(n) time
        ref: https://justindomke.wordpress.com/2009/01/17/hessian-vector-products/

        :param states: torch.Tensor(#samples, #d_state) used to calculate KL divergence on samples
        :param v: the arbitrary vector, torch.Tensor
        :return: (H + damping * I) dot v, where H = nabla nabla KL
        """
        kl = self.get_kl(states)
        # here, set create_graph=True to enable second derivative on function of this derivative
        grad_kl = torch.autograd.grad(kl, self.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grad_kl])

        grad_kl_v = (flat_grad_kl * v).sum()
        grad_grad_kl_v = torch.autograd.grad(grad_kl_v, self.parameters())
        flat_grad_grad_kl_v = torch.cat([grad.contiguous().view(-1) for grad in grad_grad_kl_v])

        return flat_grad_grad_kl_v + args.damping * v

    def set_flat_params(self, flat_params):
        """
        set flat_params

        : param flat_params: Tensor
        """
        flat_params = Tensor(flat_params)
        prev_ind = 0
        for param in self.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
            prev_ind += flat_size
        self.old_log_prob = None

    def get_flat_params(self):
        """
        get flat parameters
        returns numpy array
        """
        params = []
        for param in self.parameters():
            params.append(param.data.view(-1))
        flat_params = torch.cat(params)
        return flat_params.double().numpy()

    def save_model_policy(self):
        """
        save model
        """
        if not os.path.exists('./saved_model'):
            os.makedirs('./saved_model')
        np.save('./saved_model/param_policy_{}'.format(args.env_name), 
                self.get_flat_params())
    
    def load_model_policy(self):
        flat_params = np.load('./saved_model/param_policy_{}.npy'.format(args.env_name))
        self.set_flat_params(flat_params)




class Value(nn.Module):
    def __init__(self, num_inputs):
        super(Value, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)
        self.value_head = nn.Linear(64, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        state_values = self.value_head(x)
        return state_values.squeeze()

    def get_flat_params(self):
        """
        get flat parameters
        
        :return: flat param, numpy array
        """
        params = []
        for param in self.parameters():
            params.append(param.data.view(-1))
        flat_params = torch.cat(params)
        return flat_params.double().numpy()

    def get_flat_grad(self):
        """
        get flat gradient
        
        :return: flat grad, numpy array
        """
        grads = []
        for param in self.parameters():
            grads.append(param.grad.view(-1))

        flat_grad = torch.cat(grads)
        return flat_grad.double().numpy() 

    def set_flat_params(self, flat_params):
        """
        set flat_params
        
        :param flat_params: numpy.ndarray
        """
        flat_params = Tensor(flat_params)
        prev_ind = 0
        for param in self.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
            prev_ind += flat_size

    def reset_grad(self):
        for param in self.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

    def get_sum_squared_params(self):
        """
        sum of squared parameters used for L2 regularization
        returns a Variable
        """
        ans = Variable(Tensor([0]))
        for param in self.parameters():
            ans += param.pow(2).mean()
        return ans

    def save_model_value(self):
        """
        save model
        """
        if not os.path.exists('./saved_model'):
             os.makedirs('./saved_model')

        np.save('./saved_model/param_value_{}'.format(args.env_name), 
                self.get_flat_params())

    def load_model_value(self):
        flat_params = np.load('./saved_model/param_value_{}.npy'.format(args.env_name))
        self.set_flat_params(flat_params)
