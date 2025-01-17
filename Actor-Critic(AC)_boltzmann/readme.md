The Actor-Critic (AC) algorithm combines policy networks and value networks to address problems in both continuous and discrete action environments. It learns the optimal policy through the policy network while optimizing the value network, solving the policy optimization problem. The characteristics of the AC algorithm include utilizing complete observational information, no need for internal modeling of the environment, effectively balancing exploration and exploitation, and it is applicable to fields such as robot control and image recognition.

Basic Principle of AC Algorithm
The AC algorithm consists of two core components:

Actor: Responsible for learning and representing the policy function π(a|s). The Actor network outputs the probability distribution of action a based on the current state s.
Critic: Responsible for learning and representing the state value function V(s) or action value function Q(s,a). The Critic network predicts the expected value of cumulative discounted rewards based on the current state s and the action a taken.
The Actor network and Critic network learn interactively and optimize step by step, with the Critic network providing feedback signals to guide the Actor network to adjust the policy in a direction that can achieve higher rewards. This coupled learning approach allows the AC algorithm to maintain the convergence of policy gradient algorithms while significantly improving learning efficiency.
