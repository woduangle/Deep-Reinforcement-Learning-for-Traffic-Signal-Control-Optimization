# A3C

---

### Table of contents 
1. [Introduction](#1-introduction)
2. [Method](#2-method)
3. [Experiments](#3-experiments)
4. [Results](#4-results)
5. [Conclusions](#5-conclusions)

---
### 1. Introduction 
- A3C is a combination of the Actor-Critic framework with asynchronous training mechanisms, offering an advanced deep reinforcement learning algorithm. The core idea is to accelerate the learning process and enhance stability by having multiple parallel worker threads interact with the environment and update shared neural network parameters concurrently. Each thread independently executes policies, collects experiences, and updates weights to the global network at certain intervals. This method not only effectively avoids high variance issues common in traditional reinforcement learning but also significantly speeds up the learning process.

### 2. Method 
#### Policy gradient update rule

$$
\theta_{\text{actor}} \leftarrow \theta_{\text{actor}} + \alpha \nabla_\theta \log \pi_\theta(a|s) A(s,a)
$$

Here, α represents the learning rate, \( \pi_{\theta}(a|s) \) denotes the probability distribution of taking action \( a \) given state \( s \), and \( A(s,a) \) is the advantage function that measures how much better it is to take action \( a \) compared to average behavior.


#### Value function update rule

$$
\theta_{\text{critic}} \leftarrow \theta_{\text{critic}} + \beta (R - V_\theta(s)) \nabla_\theta V_\theta(s)
$$

Where \( \beta \) is the learning rate, \( R \) is the reward, and \( V_{\theta}(s) \) is the value function indicating the expected return starting from state \( s \) following policy.
  
### 5. Conclusions 
When applying A3C to traffic signal control within SUMO, different agents can be set for various intersections. These agents use SUMO's API to obtain real-time traffic data (such as vehicle waiting times, queue lengths, etc.) as input states. Based on this information, agents decide when to switch the traffic light colors aiming to minimize total waiting time and optimize traffic flow. SUMO allows simulating complex traffic scenarios, enabling the A3C model to train and test under conditions close to the real world.
