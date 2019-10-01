"""
TRPO: Schulman, John, et al. "Trust region policy optimization." International Conference on Machine Learning. 2015.

"""

import gym
import torch
import torch.nn as nn
import torch.optim as opt
from torch import Tensor
from torch.autograd import Variable
from collections import namedtuple
from itertools import count
import scipy.optimize as sciopt
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
from os.path import join as joindir
import pandas as pd
import numpy as np
import argparse
import datetime
import math
from model import Policy, Value
from optimizer import conjugate_gradient, line_search
from utils import *
from running_state import *

 
EPS = 1e-10
RESULT_DIR = './result'
if os.path.exists(RESULT_DIR):
    pass
else:
    os.makedirs(RESULT_DIR)

    
def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Hopper-v2', 
                        help = 'gym environment to test algorithm')
    parser.add_argument('--seed', type=int, default=64, 
                        help = 'random seed')
    parser.add_argument('--num_episode', type=int, default=200, 
                        help = '')
    parser.add_argument('--batch_size', type=int, default=5000,
                        help = '')
    parser.add_argument('--max_step_per_episode', type=int, default=200,
                        help = '')
    parser.add_argument('--gamma', type=float, default=0.995,
                        help = '')
    parser.add_argument('--lamda', type=float, default=0.97,
                        help = '')
    parser.add_argument('--l2_reg', type=float, default=1e-3,
                        help='l2 regularization regression (default: 1e-3)')
    parser.add_argument('--value_opt_max_iter', type=int, default=25,
                        help = '')
    parser.add_argument('--damping', type=float, default=0.1,
                        help = 'damping (default: 1e-1)')
    parser.add_argument('--max_kl', type=float, default=1e-2, 
                        help='max kl value (default: 1e-2)')
    parser.add_argument('--cg_nsteps', type=int, default=10,
                        help = '')
    parser.add_argument('--log_num_episode', type=int, default=1,
                        help = 'interval between training status logs (default: 1)')
    parser.add_argument('--num_parallel_run', type=int, default=5, 
                        help = 'number of asychronous training process')
    parser.add_argument('--training', type = bool, default=False, 
                        help = 'choose whether training or testing')

    args = parser.parse_args()
    return args

def select_single_action(policy_net, state):
    """
    given state returns action selected by policy_net
    select a single action used in simulation step

    :param policy_net: given policy network
    :param state: state repr, numpy.ndarray
    :return: an action repr, numpy.ndarray
    """
    state = Tensor(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(state)
    action = torch.normal(action_mean, action_std)
    return action.data[0].numpy()

def get_value_loss(flat_params, value_net, states, returns):
    """
    returns value net loss on dataset=(states, returns)
    params
        value_net
        states: Variable
        returns: Variable
    returns (loss, gradient)
    """
    value_net.set_flat_params(flat_params)
    value_net.reset_grad()

    pred_values = value_net(states)
    value_loss = (pred_values - returns).pow(2).mean() + args.l2_reg * value_net.get_sum_squared_params() 
    value_loss.backward()
    return (value_loss.data.double().numpy()[0], value_net.get_flat_grad())


def trpo(args):
    env = gym.make(args.env_name)
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    env.seed(args.seed)
    torch.manual_seed(args.seed)

    policy_net = Policy(num_inputs, num_actions)
    value_net = Value(num_inputs)
    
    running_state = ZFilter((num_inputs,), clip=5)
    running_reward = ZFilter((1,), demean=False, clip=10)
    
    reward_record = []
    global_steps = 0

    for i_episode in range(args.num_episode):
        memory = Memory()
        
        # sample data: single path method
        num_steps = 0
        while num_steps < args.batch_size:
            state = env.reset()
            state = running_state(state)
            
            reward_sum = 0
            for t in range(args.max_step_per_episode):
                action = select_single_action(policy_net, state)
                next_state, reward, done, _ = env.step(action)
                reward_sum += reward

                next_state = running_state(next_state)
                
                mask = 0 if done else 1
                
                memory.push(state, action, mask, next_state, reward)
                
                if done:
                    break
                    
                state = next_state
                
            num_steps += (t + 1)
            global_steps += (t + 1)
            reward_record.append({'steps': global_steps, 'reward': reward_sum})

        batch = memory.sample()
        batch_size = len(memory)
        
        # update params
        rewards = Tensor(batch.reward)
        masks = Tensor(batch.mask)
        actions = Tensor(batch.action)
        states = Tensor(batch.state)
        values = value_net(states)
        
        returns = Tensor(batch_size)
        deltas = Tensor(batch_size)
        advantages = Tensor(batch_size)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(batch_size)):
            returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values[i]
            # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization advantage estimate)
            # notation following PPO paper
            advantages[i] = deltas[i] + args.gamma * args.lamda * prev_advantage * masks[i]

            prev_return = returns[i]
            prev_value = values[i]
            prev_advantage = advantages[i]
        advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)
            
        # optimize value network
        loss_func_args = (value_net, states, returns)
        old_loss, _ = get_value_loss(value_net.get_flat_params(), *loss_func_args)
        flat_params, opt_loss, opt_info = sciopt.fmin_l_bfgs_b(get_value_loss, 
            value_net.get_flat_params(), args=loss_func_args, maxiter=args.value_opt_max_iter)
        value_net.set_flat_params(flat_params)
        print('ValueNet optimization: old loss = {}, new loss = {}'.format(old_loss, opt_loss))

        # optimize policy network
        # 1. find search direction for network parameter optimization, use conjugate gradient (CG)
        #       the direction can be found analytically, it s = - A^{-1} g, 
        #       where A is the Fisher Information Matrix (FIM) w.r.t. action probability distribution 
        #       and g is the gradient w.r.t. loss function \dfrac{\pi_\theta (a|s)}{q(a|s)} Q(s, a)
        policy_net.set_old_loss(states, actions)
        loss = policy_net.get_loss(states, actions, advantages)
        g = torch.autograd.grad(loss, policy_net.parameters())
        flat_g = torch.cat([grad.view(-1) for grad in g]).data
        Av = lambda v: policy_net.kl_hessian_times_vector(states, v)
        step_dir = conjugate_gradient(Av, - flat_g, nsteps=args.cg_nsteps)

        # 2. find maximum stepsize along the search direction
        #       the problem: min g * x  s.t. 1/2 * x^T * A * x <= delta
        #       can be solved analytically with x = beta * s
        #       where beta = sqrt(2 delta / s^T A s)
        sAs = 0.5 * (step_dir * Av(step_dir)).sum(0)
        beta = torch.sqrt(2 * args.max_kl / sAs)
        full_step = (beta * step_dir).data.numpy()

        # 3. do line search along the found direction, with maximum change = full_step
        #       the maximum change is restricted by the KL divergence constraint
        #       line search with backtracking method
        get_policy_loss = lambda x: policy_net.get_loss(states, actions, advantages)
        old_loss = get_policy_loss(None)
        success, new_params = line_search(policy_net, get_policy_loss, full_step, flat_g)
        policy_net.set_flat_params(new_params)
        new_loss = get_policy_loss(None)
        print('PolicyNet optimization: old loss = {}, new loss = {}'.format(old_loss, new_loss))

        if i_episode % args.log_num_episode == 0:
            print('Finished episode: {} Mean Reward: {}'.format(i_episode, reward_record[-1]))
            print('-----------------')
    
    policy_net.save_model_policy()
    value_net.save_model_value()
    return reward_record


if __name__ == "__main__":
    datestr = datetime.datetime.now().strftime('%Y-%m-%d')
    args = add_arguments()

    record_dfs = pd.DataFrame(columns=['steps', 'reward'])
    reward_cols = []
    if args.training == True:
        for i in range(args.num_parallel_run):
            args.seed += 1
            reward_record = pd.DataFrame(trpo(args))
            record_dfs = record_dfs.merge(reward_record, how='outer', on='steps', suffixes=('', '_{}'.format(i)))
            reward_cols.append('reward_{}'.format(i))
            
        record_dfs = record_dfs.drop(columns='reward').sort_values(by='steps', ascending=True).ffill().bfill()
        record_dfs['reward_mean'] = record_dfs[reward_cols].mean(axis=1)
        record_dfs['reward_std'] = record_dfs[reward_cols].std(axis=1)
        record_dfs['reward_smooth'] = record_dfs['reward_mean'].ewm(span=20).mean()
        record_dfs.to_csv(joindir(RESULT_DIR, 'trpo-record-{}-{}.csv'.format(args.env_name, datestr)))

    # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(record_dfs['steps'], record_dfs['reward_mean'], label='trajory reward')
        plt.plot(record_dfs['steps'], record_dfs['reward_smooth'], label='smoothed reward')
        plt.fill_between(record_dfs['steps'], record_dfs['reward_mean'] - record_dfs['reward_std'], 
        record_dfs['reward_mean'] + record_dfs['reward_std'], color='b', alpha=0.2)
        plt.legend()
        plt.xlabel('steps of env interaction (sample complexity)')
        plt.ylabel('average reward')
        plt.title('TRPO on {}'.format(args.env_name))
        plt.savefig(joindir(RESULT_DIR, 'trpo-{}-{}.pdf'.format(args.env_name, datestr))) 

    else:
        env = gym.make(args.env_name)
    
        num_inputs = env.observation_space.shape[0]
        num_actions = env.action_space.shape[0]

        running_state = ZFilter((num_inputs,), clip=5)
        running_reward = ZFilter((1,), demean=False, clip=10)

        env.seed(args.seed)
        torch.manual_seed(args.seed)

        policy_net = Policy(num_inputs, num_actions)
        policy_net.load_model_policy()
        
        state = env.reset()
        
        for i in range(5000):
            env.render()
            state = running_state(state)

            action = select_single_action(policy_net, state)
            next_state, reward, done, _  = env.step(action)

            #reward = running_reward(reward)
            state = next_state



            



