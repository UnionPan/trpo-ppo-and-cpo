# Intro:
This project will be aimed at reproducing some benchmark policy-search algorithms for rl control in continuous domain and exploring the connections to CMDPs, security and safety.

TRPO solves the stepsize-choosing problem of policy gradient methods by solving an approximate optimization problem iteratively(similiar to Majorization-Minimization approach), ensuring monotonic update over policy;

Following Trust Region method, PPO suggests instead of solving constraint optimization problem per iteration, simply adding penalty to the objective and thus solvable by SGD method.

Also inspired by Trust Region method, CPO is a sort of generalization of TRPO, for it directly adds constraint to the original objective, to form a linear quadratic constraint optimization problem and optimize it by solving dual problem iteratively.


# Dependencies:
<p>
 1. pytorch
<p>
 2. mujoco (student one-year free)
<p>
 3. openAI gym 
<p>
  
- [ ] garage (maybe tensorflow)
  
# Stage of work:
  
- [x] implement some preliminary algorithms based on Pytorch: TRPO and PPO
- [ ] using garage to implement the constraint policy optimization algorithm
- [ ] using garage to create some new MDPs


# References:


[1] "Trust Region Policy Optimization" 
John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel https://arxiv.org/abs/1502.05477

[2] "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel https://arxiv.org/abs/1506.02438

[3] "Proximal Policy Optimization Algorithms" 
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov https://arxiv.org/abs/1707.06347

[4] "Constrained Policy Optimization" Joshua Achiam, David Held, Aviv Tamar, Pieter Abbeel https://arxiv.org/abs/1705.10528

[5] "Benchmarking Deep Reinforcement Learning for Continuous Control" 
   Yan Duan, Xi Chen, Rein Houthooft, John Schulman, Pieter Abbeel https://arxiv.org/abs/1604.06778




