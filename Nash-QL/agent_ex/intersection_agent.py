import random
import pandas as pd
import numpy as np
import save_data.save_data as sd  # 保存数据
import net_game.net_game as ng


class IntersectionAgent:
    """
    交叉口Agent类
    Q值采用字符串实现
    """

    def __init__(self, agent_setting, rl_setting):
        """Initialization"""
        print('Agent初始化……')
        self.name = agent_setting['id']
        self.agent_id = agent_setting['cross_ids']  # agent_id denoted as cross id
        self.cross_id = self.agent_id
        self.tls_id = agent_setting['tls_ids']  # 信号灯id

        self.state_config = agent_setting['states']  # basic info about states of the agent
        self.action_config = agent_setting['actions']  # action config
        self.reward_config = agent_setting['rewards']  # reward config
        # 初始化基本参数
        state_names = self.state_config['names']
        action_names = self.action_config['names']
        self.state_num = len(state_names)
        self.action_num = len(action_names)
        self.state_names = state_names
        self.action_names = action_names
        self.action_duration = [i[2] for _, i in self.action_config['paras'].items()]  # 动作持续时间
        """网络博弈"""
        self.identity = agent_setting['identity']
        self.opponent = agent_setting['leader_opponent'] if self.identity == 'Leader' else -9999
        """强化学习"""
        print('RL learning...')
        self.learning_model = rl_setting['learning_model']  # basic parameters for RL algo
        self.action_selection_model = rl_setting['action_selection']  # action selection
        self.epsilon = rl_setting['action_selection']['epsilon']
        self.epsilon_min = rl_setting['action_selection']['epsilon_min']
        self.epsilon_decay = rl_setting['action_selection']['epsilon_decay']

        self.q_table = self.__init_q_table_from_states_and_actions(state_names, action_names)  # initialize q table
        #
        self.state_action_count_table = self.__init_state_action_count_table_from_states_and_actions(state_names,
                                                                                                     action_names)  # 状态，动作）对，被访问次数统计表
        #
        self.state_prev = ""
        self.state_current = ""
        self.action_prev = random.choice(action_names)
        self.action_current = random.choice(action_names)
        self.action_current_index = self.action_names.index(self.action_current)
        self.reward_current = 0.0
        #
        """记录当前Agent的动作类型，及当前动作的剩余时间"""
        self.current_strategy_remain_time = 0  # 当前策略剩余时间，为0时表示需要重新指定策略

    def __init_q_table_from_states_and_actions(self, state_name_list, action_name_list):
        """create q-table 仅完成初始化，通过添加index使空间动态增长"""
        q_table = pd.DataFrame(0, index=[], columns=action_name_list)
        return q_table

    def __init_state_action_count_table_from_states_and_actions(self, state_name_list, action_name_list):
        """创建状态动作访问次数统计表，用于统计状态动作对出现的次数"""
        state_action_count_table = pd.DataFrame(1, index=[], columns=action_name_list)  # 初始化值为 1
        return state_action_count_table

    def is_current_strategy_over(self):
        """判断当前动作是否执行完成"""
        if self.current_strategy_remain_time == 0:
            return True  # 当前动作执行完成
        else:
            return False

    def save_current_state(self, state_val):
        """保存当前状态，并存储上一个状态"""
        self.state_prev = self.state_current
        self.state_current = state_val

    def select_action(self):
        self.__check_state_exist(self.state_current)
        """根据当前状态选择动作，epsilon greedy"""
        action = self.epsilon_greedy(epsilon=self.epsilon, state=self.state_current, q_table=self.q_table,
                                     action_num=self.action_num)
        # 保存动作，并设置计时器
        self.save_action_set_time_counter(action_index=action)
        # 调整epsilon
        self.epsilon_decaying()

    def epsilon_greedy(self, epsilon, state, q_table, action_num):
        """epsilon greedy"""
        if np.random.uniform() < epsilon:  # 选择Q_value 最高的action
            # choose random action
            action_index = np.random.choice(action_num)
        else:
            # choose best action
            state_action = q_table.loc[state, :]  # loc是获取一列的值:取q_table的observation行，所有列
            # some actions may have the same value, randomly choose on in these actions
            action_name = np.random.choice(
                state_action[state_action == np.max(state_action)].index)  # np.max（）：取行方向的最大值
            action_index = list(state_action.index).index(action_name)  # 找到动作对应的index
        return action_index

    def save_action_set_time_counter(self, action_index):
        """保存动作索引、名称；设置计时器"""
        self.action_current_index = action_index
        # 设置动作执行计时器，剩余时间
        self.current_strategy_remain_time = self.action_duration[action_index]

    def time_count_step(self):
        """将Agent的策略执行时间减1，表示执行1个步长"""
        if self.current_strategy_remain_time > 0:
            self.current_strategy_remain_time -= 1
        else:  # 恢复原始状态
            self.current_strategy_remain_time = 0

    def save_reward(self, val):
        """保存当前的奖励"""
        print(self.agent_id, '-REWARD:', val)
        self.reward_current = val
        sd.save_reward(self.agent_id, val)  # 保存本次奖励值a

    def __check_state_exist(self, state):
        """检查state是否存在q table中"""
        q_table_idx = self.q_table.index
        if q_table_idx.isin([state]).any():
            pass
        else:
            df = pd.DataFrame(0, index=[state], columns=self.action_names)
            self.q_table = pd.concat([self.q_table, df])  # 将df合并

    def __check_state_exist_in_count_table(self, state):
        """在统计表中检查状态是否存在"""
        q_table_idx = self.state_action_count_table.index
        if q_table_idx.isin([state]).any():
            pass
        else:
            df = pd.DataFrame(1, index=[state], columns=self.action_names)  # 初始化为1次
            self.state_action_count_table = pd.concat([self.state_action_count_table, df])  # 将df合并

    def get_q_value(self, state=-99, action=-99):
        state = self.state_prev
        action = self.action_current
        """获取Q值"""
        self.__check_state_exist(state)
        q_value = self.q_table.loc[state, action]
        return self.q_table.loc[state, action]

    def update_q_table_ql_single(self):
        """更新Q表，只考虑本地信息"""
        self.__check_state_exist(self.state_prev)
        self.__check_state_exist(self.state_current)
        # 获取参数
        alpha = self.learning_model['learning_rate']
        gamma = self.learning_model['gamma']
        #
        pre_q = self.q_table.loc[self.state_prev, self.action_current]  # 获取上一个Q值
        q_max_for_post_state = self.q_table.loc[self.state_current, :].max()  # 获取当前状态的最大Q值
        # QL算法Q值更新公式
        if self.identity == 'Leader':
            q_new = pre_q * (1 - alpha) + alpha * (self.reward_current + gamma * ng.nash_q_values[self.name])
            #  ng.nash_q_values[self.name] 是Nash 均衡 Q 值，通过查找 ng.nash_q_values 字典得到
        elif self.identity == 'Follower':
            q_new = pre_q * (1 - alpha) + alpha * (self.reward_current + gamma * q_max_for_post_state)
        else:
            raise ValueError("identity类型错误")
        # 将Q值更新至Q表中
        self.q_table.loc[self.state_prev, self.action_current] = q_new

    def epsilon_decaying(self):
        # 调整epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    '''------------------------------------------------------'''
