import random
import pandas as pd
import numpy as np
import save_data.save_data as sd # Save data


def boltzmann(**kwargs):
    """Boltzmann or Softmax equation"""
    state = kwargs['state']
    q_table = kwargs['q_table']
    temperature = kwargs['temperature']
    #
    q_state_action = q_table.loc[state, :]  # loc is used to get the values of a column: take the observation row of q_table, all columns
    e_q_state_action = np.exp(q_state_action / temperature)  #
    e_q_state_action_sum = np.sum(e_q_state_action)
    #
    probability_actions = np.true_divide(e_q_state_action, e_q_state_action_sum)
    #
    action_selected = np.random.choice(
        q_state_action[(q_state_action == np.random.choice(q_state_action, p=probability_actions))].index)
    return action_selected


class IntersectionAgent:
    """
    Intersection Agent class
    Q-values are implemented using strings
    """

    def __init__(self, agent_setting, rl_setting):
        """Initialization"""
        print('Agent initialization...')
        self.agent_id = agent_setting['cross_ids']  # agent_id denoted as cross id
        self.cross_id = self.agent_id
        self.tls_id = agent_setting['tls_ids']  # Traffic light ID
        self.state_config = agent_setting['states']  # basic info about states of the agent
        self.action_config = agent_setting['actions']  # action config
        self.reward_config = agent_setting['rewards']  # reward config
        # Initialize basic parameters
        state_names = self.state_config['names']
        action_names = self.action_config['names']
        self.state_num = len(state_names)
        self.action_num = len(action_names)
        self.state_names = state_names
        self.action_names = action_names
        self.action_duration = [i[2] for _, i in self.action_config['paras'].items()]  # Duration of action
        """Initialization"""
        print('RL learning...')
        self.learning_model = rl_setting['learning_model']  # basic parameters for RL algo
        self.action_selection_model = rl_setting['action_selection']  # action selection
        #
        self.q_table = self.__init_q_table_from_states_and_actions(state_names, action_names)  # initialize q table
        #
        self.state_action_count_table = self.__init_state_action_count_table_from_states_and_actions(state_names,
                                                                                                     action_names)  # State-action pair access count table
        #
        self.state_prev = ""
        self.state_current = ""
        self.action_prev = random.choice(action_names)
        self.action_current = random.choice(action_names)
        self.action_current_index = self.action_names.index(self.action_current)
        self.reward_current = 0.0
        #
        """Record the current Agent's action type and the remaining time of the current action"""
        self.current_strategy_remain_time = 0  # Remaining time for the current strategy; when it is 0, it indicates that a new strategy needs to be specified

    def __init_q_table_from_states_and_actions(self, state_name_list, action_name_list):
        """Create q-table, only complete initialization, dynamically grow the space by adding indexes"""
        q_table = pd.DataFrame(0, index=[], columns=action_name_list)
        return q_table

    def __init_state_action_count_table_from_states_and_actions(self, state_name_list, action_name_list):
        """Create a state-action access count table to track the number of occurrences of state-action pairs"""
        state_action_count_table = pd.DataFrame(1, index=[], columns=action_name_list)  # Initialize the value to 1
        return state_action_count_table

    def is_current_strategy_over(self):
        """Determine if the current action has been executed"""
        if self.current_strategy_remain_time == 0:
            return True  # The current action is executed
        else:
            return False

    def save_current_state(self, state_val):
        """Save the current state and store the previous state"""
        self.state_prev = self.state_current
        self.state_current = state_val

    def select_action(self):
        """Boltzmann or Softmax method"""
        #
        self.__check_state_exist(self.state_current)
        # Retrieve parameters
        temperature = self.action_selection_model['temperature']
        # Call the calculation function
        action_selected = boltzmann(state=self.state_current, q_table=self.q_table, temperature=temperature)
        action_index = self.action_names.index(action_selected)  # Find the index corresponding to the action
        # Save the action and set the timer
        self.save_action_set_time_counter(action_index)

    def save_action_set_time_counter(self, action_index):
        """Save the action index and name; set the timer"""
        self.action_current_index = action_index
        # Set the action execution timer, remaining time
        self.current_strategy_remain_time = self.action_duration[action_index]

    def time_count_step(self):
        """Decrease the Agent's strategy execution time by 1, indicating the execution of one time step"""
        if self.current_strategy_remain_time > 0:
            self.current_strategy_remain_time -= 1
        else:  # Restore the original state
            self.current_strategy_remain_time = 0

    def save_reward(self, val):
        """Save the current reward"""
        print(self.agent_id, '-REWARD:', val)
        self.reward_current = val
        sd.save_reward(self.agent_id, val)  # Save the current reward value a

    def __check_state_exist(self, state):
        """Check if the state exists in the q-table"""
        q_table_idx = self.q_table.index
        if q_table_idx.isin([state]).any():
            pass
        else:
            df = pd.DataFrame(0, index=[state], columns=self.action_names)
            self.q_table = pd.concat([self.q_table, df])  # Merge df

    def __check_state_exist_in_count_table(self, state):
        """Check if the state exists in the count table"""
        q_table_idx = self.state_action_count_table.index
        if q_table_idx.isin([state]).any():
            pass
        else:
            df = pd.DataFrame(1, index=[state], columns=self.action_names)  # Initialize as 1 occurrence
            self.state_action_count_table = pd.concat([self.state_action_count_table, df])  # Merge df

    def update_q_table_actor_critic(self):
        """Update the Q-table, considering only local information"""
        self.__check_state_exist(self.state_prev)
        self.__check_state_exist(self.state_current)
        # Retrieve parameters
        alpha = self.learning_model['learning_rate']
        gamma = self.learning_model['gamma']
        #
        prev_q = self.q_table.loc[self.state_prev, self.action_current]  # Obtain the previous Q-value
        prev_q_max = self.q_table.loc[self.state_prev, :].max()  # Obtain the maximum Q-value of the previous state
        current_q_max = self.q_table.loc[self.state_current, :].max()  # Obtain the maximum Q-value of the current state
        # QL algorithm Q-value update formula
        q_new = prev_q + alpha * (self.reward_current + current_q_max - prev_q_max)
        # 将Update the Q-value in the Q-table
        self.q_table.loc[self.state_prev, self.action_current] = q_new

