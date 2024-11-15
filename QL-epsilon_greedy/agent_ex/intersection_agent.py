import random
import pandas as pd
import numpy as np
import save_data.save_data as sd # 保存数据

class IntersectionAgent:
    """
    Intersection Agent Class
    Q-value is implemented using strings
    """

    def __init__(self, agent_setting, rl_setting):
        """initialization"""
        print('agent initialization……')
        self.agent_id = agent_setting['cross_ids']  # agent_id denoted as cross id
        self.cross_id = self.agent_id
        self.tls_id = agent_setting['tls_ids']  # signal lamp id
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
        self.action_duration = [i[2] for _, i in self.action_config['paras'].items()]  # action duration
        """Initialization"""
        print('RL learning...')
        self.learning_model = rl_setting['learning_model']  # basic parameters for RL algo
        self.action_selection_model = rl_setting['action_selection']  # action selection
        self.epsilon = rl_setting['action_selection']['epsilon']
        self.epsilon_min = rl_setting['action_selection']['epsilon_min']
        self.epsilon_decay = rl_setting['action_selection']['epsilon_decay']
        #
        self.q_table = self.__init_q_table_from_states_and_actions(state_names, action_names)  # initialize q table
        #
        self.state_action_count_table = self.__init_state_action_count_table_from_states_and_actions(state_names,
                                                                                                     action_names)  # status, action）answer，statistics of visited times
        #
        self.state_prev = ""
        self.state_current = ""
        self.action_prev = random.choice(action_names)
        self.action_current = random.choice(action_names)
        self.action_current_index = self.action_names.index(self.action_current)
        self.reward_current = 0.0
        #
        """record the action type of the current agent and the remaining time of the current action"""
        self.current_strategy_remain_time = 0  # when the remaining time of the current policy is 0, it means that a new policy needs to be specified

    def __init_q_table_from_states_and_actions(self, state_name_list, action_name_list):
        """create q-table only complete initialization, dynamically increase space by adding index"""
        q_table = pd.DataFrame(0, index=[], columns=action_name_list)
        return q_table

    def __init_state_action_count_table_from_states_and_actions(self, state_name_list, action_name_list):
        """create a status action access count table to track the frequency of occurrence of status action pairs"""
        state_action_count_table = pd.DataFrame(1, index=[], columns=action_name_list)  # the initialization value is 1
        return state_action_count_table

    def is_current_strategy_over(self):
        """determine whether the current action has been completed"""
        if self.current_strategy_remain_time == 0:
            return True  # current action execution completed
        else:
            return False

    def save_current_state(self, state_val):
        """save the current state and store the previous state"""
        self.state_prev = self.state_current
        self.state_current = state_val

    def select_action(self):
        """select an action based on the current state，epsilon greedy"""
        self.__check_state_exist(self.state_current)
        action = self.epsilon_greedy(epsilon=self.epsilon,state=self.state_current,q_table=self.q_table,action_num=self.action_num)
        # save action and set timer
        self.save_action_set_time_counter(action_index=action)
        # adjust epsilon
        self.epsilon_decaying()

    def epsilon_greedy(self,epsilon, state, q_table, action_num):
        """epsilon greedy"""
        if np.random.uniform() < epsilon:  # chose Q_value the highest action
            # choose random action
            action_index = np.random.choice(action_num)
        else:
            # choose best action
            state_action = q_table.loc[state, :]  # lOC is to obtain the value of a column: take the observation row of q_table and all columns
            # some actions may have the same value, randomly choose on in these actions
            action_name = np.random.choice(state_action[state_action == np.max(state_action)].index)  # np.max（）：take the maximum value in the row direction
            action_index = list(state_action.index).index(action_name)  # find the index corresponding to the action
        return action_index

    def save_action_set_time_counter(self, action_index):
        """save action index and name; set timer"""
        self.action_current_index = action_index
        # set action execution timer, remaining time
        self.current_strategy_remain_time = self.action_duration[action_index]

    def time_count_step(self):
        """subtracting 1 from the execution time of the agent's strategy means executing 1 step"""
        if self.current_strategy_remain_time > 0:
            self.current_strategy_remain_time -= 1
        else:  # restore the original state
            self.current_strategy_remain_time = 0

    def save_reward(self, val):
        """save the current reward"""
        print(self.agent_id, '-REWARD:', val)
        self.reward_current = val
        sd.save_reward(self.agent_id, val)  # save this reward value a

    def __check_state_exist(self, state):
        """check if the state exists in the q table"""
        q_table_idx = self.q_table.index
        if q_table_idx.isin([state]).any():
            pass
        else:
            df = pd.DataFrame(0, index=[state], columns=self.action_names)
            self.q_table = pd.concat([self.q_table, df])  # merge df

    def __check_state_exist_in_count_table(self, state):
        """check if the status exists in the statistical table"""
        q_table_idx = self.state_action_count_table.index
        if q_table_idx.isin([state]).any():
            pass
        else:
            df = pd.DataFrame(1, index=[state], columns=self.action_names)  # initialize once
            self.state_action_count_table = pd.concat([self.state_action_count_table, df])  # merge df

    def update_q_table_ql_single(self):
        """update Q table to only consider local information"""
        self.__check_state_exist(self.state_prev)
        self.__check_state_exist(self.state_current)
        # get parameters
        alpha = self.learning_model['learning_rate']
        gamma = self.learning_model['gamma']
        #
        pre_q = self.q_table.loc[self.state_prev, self.action_current]  # get the previous Q value
        q_max_for_post_state = self.q_table.loc[self.state_current, :].max()  # obtain the maximum Q value of the current state
        # Q-value update formula for QL algorithm
        q_new = pre_q * (1 - alpha) + alpha * (self.reward_current + gamma * q_max_for_post_state)
        # update the Q value to the Q table
        self.q_table.loc[self.state_prev, self.action_current] = q_new

    def epsilon_decaying(self):
        # adjust epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay