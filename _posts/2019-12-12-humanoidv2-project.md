---
layout: post
title: Make a three-dimensional bipedal robot walk
---
## Objective
Learn & Apply reinforcement learning techniques on complex continuous control domain to achieve maximum rewards. In the continuous control domain, where actions are continuous and often high-dimensional such as OpenAI-Gym environment Humanoid-V2. The Humanoid environment has 377 Observation dimensions and 17 action dimensions. This problem requires temporal difference learning compared to supervised learning since it has so many moving parts that are hard to debug, and they require substantial efforts in tuning in order to get good results. Also, in supervised learning problems, progress has been driven by large labeled datasets like ImageNet. In Reinforcement Learning, the closest equivalent would be a large and diverse collection of environments.

## Dataset
In supervised learning, labeled datasets have been used to train the model. In Reinforcement learning, the closest equivalent would be a large and diverse collection of environments. Typically, the environment is a set of states the agent is attempting to influence via its choice of actions. (rphv, 2016) Environment is a modeled as a stochastic finite machine with inputs and outputs. Inputs are actions sent to the agents, Outputs are observations and rewards sent to the agent. (Pomdp @ Www.Cs.Ubc.Ca, n.d.). The structure of the environment is depending on determining what signals are relevant and how they interact is highly specific to the problem at hand. The agent-environment boundary can be located at different purposes. In practice, the agent-environment boundary is determined once has selected particular states, actions, and rewards, and thus has identified a specific decision-making task of interest.
![image](/public/images/humanoid/environment.png)
Beyond the agent, and the environment, there are three main sub-elements of a reinforcement learning system. Policy, Reward, and Value function. A policy defines the behavior of the agent, in fact it is a brain that operates the agent. Policy can be as simple as a look up table of states and the action to be taken from the state. In general, policies may be stochastic. A reward signal is an output of environment-agent upon performing an action. It is a goal of an agent in reinforcement learning to achieve maximum reward. If low reward is achieved, then the policy would be changed to select other actions. The reward signal is the primary basis for altering the policy. Value function defines the accumulated rewards for trajectory or series of states to indicate long-term desirability. Rewards are in a sense primary, whereas values, as predictions of rewards, are secondary. Without rewards, there could be no values, and the only purpose of estimating values is to achieve more rewards. Action choices are made based on the value functions. The value functions estimate states of highest values. Rewards are direct and immediate outputs from the environment, whereas Values are to be estimated and re-estimated over a sequence of observations an agent makes over its lifetime.(Sutton & Barto, 1951) Subtle differences in the problem definition, such as the reward function or the set of actions, can drastically alter a task’s difficulty. The issue makes it difficult to reproduce published research and compare results from different papers. In this project, I am going to use OpenAI with Gym environments. Specifically, I am going to use Humanoid-V2 environment with the goal of making the Humanoid Robot walk using Proximity Policy Optimization algorithms. OpenAI Gym is a toolkit for developing and comparing reinforcement learning algorithms. It makes no assumption about the structure of your agent, and is compatible with any numerical computation libraries, such as TensorFlow or Theano. The simulations are carried out using MuJoCo, which stands for Multi Joint dynamics using Contacts. It is a platform independent physics simulator tailored to control applications. Multi-joint dynamics are represented in joint coordinates and computed via recursive algorithms. The computation is O(n3) because the inverse inertia matrix is needed to compute contact responses, however due to tree-induced sparsity, performance is comparable to O(n) algorithms in typical usage scenarios. The humanoid model has 16 joints, it is 1.6m tall and weighs 55kg. It is also modeled with 22 DoF. It amounts to 17 different actuators (actions) and 376 observations. (Tassa, Erez, & Todorov, 2012)

|Actuators|
|---|
|Abdomen_y|
|Abdomen_x|
|Abdomen_z|
|Right_hip_x|
|Right_hip_y|
|Right_hip_z|
|Right_knee|
|Left_hip_x|
|Left_hip_y|
|Left_hip_z|
|Left_knee|
|Right_shoulder1|
|Right_shoulder2|
|Right_elbow|
|Left_shoulder1|
|Left_shoulder2|
|Left_elbow|

## Observations & Action
{% highlight python %}
def exploreEnvironment(env_name):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print("Observations dimension: "+str(obs_dim))
    print("Actions dimension: "+str(act_dim))
    print("Observations: "+str(env.observation_space.sample()))
    print("Actions: "+str(env.action_space.sample()))
exploreEnvironment("Humanoid-v2")
{% endhighlight %}

```
Observations dimension: 376
Actions dimension: 17
Observations: [-9.35970989e-01 -3.80436375e-01  5.25278005e-01 -1.83726225e-01
  9.49598760e-01 -7.38980347e-01 -1.85619784e+00  3.67856896e-01
 -2.79334073e+00 -1.49212202e-01  4.86305928e-01 -7.06050332e-01
 -1.05450111e-01 -9.38980543e-01 -1.70661850e+00 -6.58521275e-01
 -5.11633139e-01 -8.20424219e-01 -1.99117451e-01  1.36231591e+00
  9.72149439e-01  2.18460133e-02  1.67010333e+00 -6.70406541e-01
 -1.00377727e+00  4.87959407e-01 -4.60841635e-01  1.53054735e+00
  2.22813119e-01 -2.02195459e-01 -6.02480876e-01 -1.39918518e+00
 -1.13670389e+00 -1.34794428e+00 -5.48158990e-01 -5.97868590e-01
 -9.41014270e-01 -4.44938763e-01  9.25651870e-02  1.17877591e+00
 -6.28102060e-01 -1.83073308e-01 -1.22353662e+00 -2.50195778e-01
 -1.12400572e+00 -6.78970923e-01  1.85790463e+00  1.97680981e+00
  5.32797070e-01  5.75083146e-01  3.27227999e-01 -3.24129450e-01
 -1.13786351e+00 -1.34016576e+00 -4.60020717e-01  6.95215761e-01
 -1.70870866e-01  4.17656717e-01  4.08197553e-01 -6.14429669e-01
  1.18112842e-01 -7.02480704e-02  3.19350844e-01  1.01002884e+00
 -2.77134767e-01  6.32999028e-02  2.48501996e-01 -2.73742433e-01
 -1.59030494e+00 -1.29834168e+00 -1.52399144e-01  2.85346669e-01
  2.36102868e+00 -5.19096963e-01 -1.20849899e+00 -8.26989957e-01
 -3.86660516e-01  1.05656656e+00  8.43768296e-02  5.10679175e-01
 -5.13757336e-01  1.05542484e+00  1.07175401e+00  3.07243196e-01
  2.66110823e-01 -3.44663667e-01  9.78117903e-01 -3.75898979e-01
 -1.78559093e+00 -1.21480213e+00 -5.65244388e-01  4.93128882e-01
 -1.77697385e+00 -2.79506509e-03  1.77468937e+00 -1.76476855e+00
  5.51900786e-02 -1.19827831e+00  1.65259015e+00 -4.84673504e-01
 -9.74177553e-01 -8.28405490e-01 -8.23419659e-01  2.40578974e+00
 -2.78097012e-01  1.17048241e+00 -4.37523206e-01  1.24646418e+00
  7.21199163e-01 -7.32044934e-04  1.70051263e+00  8.03630577e-01
 -1.49718176e-01 -3.12414726e-01  3.14491835e-01 -8.60762171e-01]
Actions: [-0.14736511 -0.19623487  0.29322252  0.15847471 -0.365481    0.1199538
 -0.17732935  0.39098296  0.17173615  0.01764819 -0.03346028  0.25502512
 -0.3731483   0.11798568  0.34848246  0.36746952 -0.1326676 ]
```

The values are high-dimensional numerical datasets. The desired behavior is assessed by the reward gained for every step. Reward is calculated by various metrics calculated from body mass, quad_control_cost, quad_impact_cost, lin_vel_cost and alive_bonus.

## PPO Algorithm
There are several approaches to the Reinforcement Learning with Neural Network function approximators such as deep Q-learning, Vanilla policy gradient methods, and trust region/ natural policy gradient methods. However, these approaches have their drawbacks. For example, deep Q-learning is often poorly understood, complex in nature. Vanilla policy gradient methods have poor data efficiency and robustness. Trust region policy optimization is relatively complicated and is not compatible with architectures that include noise or parameter sharing. Proximity Policy Optimization (PPO) is proposed to achieve the data efficiency and reliable performance. This proposed method alternates between sampling data through interaction with the environment and optimizing a surrogate objective function using stochastic gradient ascent. Standard policy gradient methods perform one gradient update per data sample, PPO performs multiple epochs of minibatch updates. Further TRPO has a constraint to stay near the old policy has achieved excellent results in continuous control tasks. Any large change from the previous policy in high dimensional, nonlinear environment leads to drastic performance issues can even result in erratic learning of the agent. One way to achieve it is by taking micro-level policy steps by controlling the learning rate. However, learning rate & policy step size depend on many factors such as model architecture, optimizer algorithm, number of training epochs and data. In PPO the TRPO constraint is achieved through training loss function. With this loss function in place, policy is trained using neural network.

### Implementation of Policy orchestration.

{% highlight python %}
Loop:
  States, actions, rewards = Run_episode(policy);
  Values = value_function(states);
  Advantage = calculate_advantage(rewards, Values);
  Newdata = Concatenate(States, actions, rewards, values, advantage);
EndLoop;
Policy.update(Newdata);
Valuation_function.retrain(data+Newdata)
{% endhighlight %}

Every time when an episode is executed with the present policy, observations such as States & actions are collected along with the rewards. The States are further used to predict the values. The values & rewards are used to calculate the advantage. With the new data the policy is updated to reflect the new learnings. Also the value function is retrained using new dataset.

### Implementation of Policy

Policy function is implemented with 3-layer Neural Network. The networks are sized based on the number of observation and action dimension. Hidden layers use tanh activation.  Hidden layer sizes are determined by observation dimension & action dimension. Hidden layer 2 is a geometric mean of hidden layer 1 and hidden layer 3.

{% highlight python %}
hid1_size = self.obs_dim * self.hid1_mult  
hid3_size = self.act_dim * 10  
hid2_size = int(np.sqrt(hid1_size * hid3_size))
{% endhighlight %}

There are three loss functions defined. Loss function 1 is standard policy gradient function. Loss function 2 is KL Divergence {D_KL(pi_old / pi_new)} and finally Loss function 3 is Hinge loss to be used when D_KL exceeds the target value.  Finally the NN use AdamOptimizer minimizing the overall loss. Also, after every episode the policy is getting updated based on the D_KL and KL_target by adjusting the learning rate.

## Value Function

Value functions are used to understand the value of a state, or state-action pair. Value is considered long term reward if we start in a state or state-action pair. Value function is Neural network with three hidden layers. The network is sized based on observation dimension. It uses Adam Optimizer which uses squared loss function. The value function used to predict a value from the state or state-action pair information. The value is discounted sum of rewards from a trajectory. During this process, the Advantage is also calculated to feed into the policy function (loss calculations). However, Advantage is calculated using Generalized Advantage Estimator instead of discounted sum of rewards.

## Training

I trained the network with adam optimizer on my Macbook Pro. It took me 48 hrs to finish 60,000 episodes. I used the default batch size=20, kl target value is 0.003, GAE is 0.98, discount factor is 0.995 and learning rate is 0.0009. I used the same parameters and network code from Patrick Coady.

## Performance Evaluation

The agent is allowed to train for 60,000 episodes. Figure 2 shows the how each episode is helping to reach the desired reward threshold. Also, Figure 3 shows the KL divergence is kept under the predefined threshold. There are other visualization that represents metrics like mean observation statistics, policy entropy and etc.

![image](/public/images/humanoid/meanreward.png)
Figure 2: Mean Reward

![image](/public/images/humanoid/kl.png)
Figure 3: D_KL(pi_old||pi_new)

![image](/public/images/humanoid/valfun.png)
Figure 4: Value Function Explained Variance

![image](/public/images/humanoid/policyentropy.png)
Figure 5: Policy Entropy

![image](/public/images/humanoid/obs.png)
Figure 6: Observation Statistics


## Video
[![training](http://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/jWmjks9W8wQ)

## Further Discussions
These models further can be evaluated for transfer learning. Such as pre-trained humanoid stand up, walking models can be used for future applications such as sit-down, stair claiming and etc. (Coady, 2017)

## References
Coady, P. (2017). AI Gym Workout.
Pomdp @ Www.Cs.Ubc.Ca. (n.d.). Retrieved from https://www.cs.ubc.ca/~murphyk/Bayes/pomdp.html
rphv. (2016). how-is-the-environment-designed-for-testing-a-reinforcement-learning-algorithm @ cs.stackexchange.com. Retrieved from https://cs.stackexchange.com/questions/56644/how-is-the-environment-designed-for-testing-a-reinforcement-learning-algorithm
Sutton, R. S., & Barto, A. G. (1951). Reinforcement Learning: An Introduction. The Lancet, 258(6685), 675–676. https://doi.org/10.1016/S0140-6736(51)92942-X
Tassa, Y., Erez, T., & Todorov, E. (2012). Synthesis and stabilization of complex behaviors through online trajectory optimization. IEEE International Conference on Intelligent Robots and Systems, 4906–4913. https://doi.org/10.1109/IROS.2012.6386025
