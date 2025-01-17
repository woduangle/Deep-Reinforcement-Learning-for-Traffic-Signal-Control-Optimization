import pandas as pd
import numpy as np
import os
import random
from collections import deque
import tensorflow as tf
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout, SimpleRNN, LSTM, GRU, Input
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical, plot_model  # 使用得到神经网络模型参数个数....
import save_data.save_data as sd  # 保存数据
import copy


class MFQMLP_TLSAgent:
    """
    交叉口信号控制器Agent类
    """

    def __init__(self, *args):
        agent_setting = args[0]  # 状态、动作、奖励设施 AGENT_ELEMENT_SETTING
        hyper_setting = args[1]  # 超参数 HYPERPARAMETER_SETTING
        """Initialization"""
        print('Agent初始化……')
        self.type = 'TLS'
        # agent_id denoted as cross id
        self.name = agent_setting['id']  # agent名称
        self.agent_id = agent_setting['id']
        self.cross_id = self.agent_id
        self.tls_id = agent_setting['id']  # 信号灯id
        # basic info about states of the agent
        self.state_config = agent_setting['states']  # agent状态设置
        self.action_config = agent_setting['actions']  # agent动作设置
        self.reward_config = agent_setting['rewards']  # agent奖励设置
        self.neighbor = agent_setting['neighbor']  # agent邻居设置
        #
        """初始化基本参数"""
        self.state_names = self.state_config['names']  # 状态类型名称
        self.action_names = self.action_config['names']  # 动作类型名称
        self.state_num = len(self.state_names)  # 状态数量
        self.state_size = np.size(self.state_names)  # 状态空间大小
        self.action_num = len(self.action_names)  # 动作数量
        self.action_size = np.size(self.action_names)  # 动作空间大小
        self.action_duration = [i[2] for _, i in self.action_config['paras'].items()]  # 动作持续时间
        # 初始值
        self.state_prev = [0] * self.state_num  # 得到全为0的空的动作空间——初始值设定
        self.state_current = [0] * self.state_num  # 得到全为0的空的动作空间
        self.action_current = random.choice(self.action_names)  # 初始随机选择动作
        self.action_current_index = self.action_names.index(self.action_current)  # 得到当前动作对应的序号
        self.reward_current = 0.0
        self.nbr_action_probs = [0] * self.action_size  # 所有邻居的动作概率分布

        """记录当前Agent的动作类型，及当前动作的剩余时间"""
        self.current_strategy_remain_time = 0  # 当前策略剩余时间，为0时表示需要重新指定策略
        """动作选择：参数"""
        # self.action_selection = hyper_setting['action_selection']['model']  # 动作选择模型
        self.decay_rate = hyper_setting['action_selection']['decay_rate']  #
        self.temperature = hyper_setting['action_selection']['temperature']  # 温度系数
        self.current_t = hyper_setting['action_selection']['current_t']
        """深度学习超参数"""
        self.mlp_type = hyper_setting['MLP_model']['type']  # 神经网络模型？
        self.input_dim = self.action_num
        self.input_shape = (self.action_num, 1)
        self.units = hyper_setting['MLP_model']['units']  # 神经元数量
        self.batch_size = hyper_setting['MLP_model']['batch_size']  # 神经元批量大小
        #
        self.learning_model = hyper_setting['learning_model']['model']  # 强化学习模型
        self.learning_rate = hyper_setting['learning_model']['learning_rate']  # 学习因子
        self.gamma = hyper_setting['learning_model']['gamma']  # 折扣因子
        self.maxlen = hyper_setting['learning_model']['maxlen']  #
        """MLP model"""
        self.model, self.target_model = self.neural_network(self.name)  # 神经网络？
        """replay memory"""
        self.reply_memory = deque(maxlen=self.maxlen)  # 双向队列

        self.__init_q_table_from_states_and_actions(self.name)  # initialize q table

    def __init_q_table_from_states_and_actions(self, name):
        """create q-table 仅完成初始化，通过添加index使空间动态增长"""
        if os.path.exists(name + 'MFD-QL QTable.csv'):  # 判断括号里的文件是否存在
            self.Q_Table = pd.read_csv(name + 'MFD-QL QTable.csv', index_col=0)
        else:
            self.Q_Table = pd.DataFrame(columns=self.action_names, dtype=np.float64)

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

    def check_state_exist(self, state):
        """检查state是否存在q table中"""
        q_table_idx = self.Q_Table.index
        if str(state) not in q_table_idx:
            self.Q_Table = pd.concat([
                self.Q_Table,
                pd.DataFrame(
                    [[0] * self.action_size],
                    index=[str(state)],
                    columns=self.Q_Table.columns
                )
            ])

    def get_nbrs(self):
        return self.neighbor

    def save_nbr_action_probs(self, nbr_action_probs):
        self.nbr_action_probs = nbr_action_probs

    def get_q_value(self, state, action):
        self.check_state_exist(state)
        return self.Q_Table.loc[state, action]

    def select_action(self):
        """选择动作"""


        '''当前agent的动作选择概率并完成动作选择'''
        self.check_state_exist(self.state_current)
        action_probs_numes = []
        denom = 0
        for i in self.action_names:
            try:
                val = np.exp(self.Q_Table.loc[str(self.state_current), i] / self.current_t)
            except OverflowError:
                return i
            action_probs_numes.append(val)
            denom += val
        action_prob = [x / denom for x in action_probs_numes]  # 得到当前agent的所有动作选择概率
        action = np.random.choice(self.action_num, 1, p=action_prob)  # 按得到的概率进行随机选取动作
        # 保存动作，并设置计时器
        self.save_action_set_time_counter(action_index=action)

    def save_action_set_time_counter(self, action_index):
        """保存动作索引、名称；设置计时器"""
        action_index = int(action_index)
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
        print(self.name, '-REWARD:', val)
        self.reward_current = val
        sd.save_reward(self.name, val)  # 保存本次奖励值

    """
    以下为深度神经网络相关函数
    """

    def neural_network(self, name):
        """构造新的神经网络，或者读入保存的神经网络"""
        if os.path.exists(name + 'model.h5'):
            model = self.load_neural_networks(name=name, path='model.h5')
            target_model = self.load_neural_networks(
                name=name, path='target model.h5')

        else:  # 构造新的dnn
            model = self.build_mlp()
            target_model = self.build_mlp()

        return model, target_model

    def model_predict(self, state):
        """模型预测"""
        # print(self.model.input_shape)
        state = np.expand_dims(state, axis=0)  # 扩维
        output = self.model.predict(state)
        return output

    def load_neural_networks(self, name, path):
        """加载已保存的model"""
        return tf.keras.models.load_model(name + path)

    def build_mlp(self):
        # 参数赋值
        lr = self.learning_rate
        input_dim = self.input_dim
        output_dim = self.action_size
        units = self.units
        """创建mlp"""
        # 创建神经网络，设置参数
        model = Sequential()
        model.add(Dense(units=units, input_dim=input_dim + 1, activation='relu'))
        model.add(Dense(units=units, activation='relu'))
        model.add(Dense(output_dim, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=lr))
        return model

    def memorize(self, done=False):
        """保存<s,a,r,s'>到经验池"""
        self.reply_memory.append(
            (self.state_prev, self.action_current_index, self.reward_current, self.state_current, self.nbr_action_probs,
             done))

    def experience_replay(self):
        """经验回放"""
        gamma = self.gamma

        try:
            # 采样方法一：此方法在batch-size超过已有数量时，会产生异常
            mini_batch = random.sample(self.reply_memory, self.batch_size)  # 在经验池中取出一些数值进行经验回放
        except (ValueError):
            # 异常处理，在数量不够时，不进行之后的操作
            return -1

        state_f, target_f = [], []  # for training

        for state, action_index, reward, next_state, action_probs, done in mini_batch:
            # next_state = np.expand_dims(next_state, axis=0)  # 扩维
            # state = np.expand_dims(state, axis=0)  # 扩维
            if len(action_probs) == 10:  # 2024.4.7未解决bug——action_probs随机凭空多一个元素，暂时进行强制删除
                action_probs.pop(-1)
            self.check_state_exist(state)
            action_probs_ = copy.deepcopy(action_probs)
            Q_v = self.Q_Table.loc[str(state), self.action_names[action_index]]  # 当前状态的所有动作的Q值
            Q_v_ = self.Q_Table.loc[str(next_state), self.action_names[action_index]]  # 下一个状态的所有动作的Q值
            action_probs.append(Q_v)  # 动作概率的list增加Q值
            action_probs_.append(Q_v_)
            model_train = np.reshape(action_probs, [1, self.action_num+1])  # 更改动作选择概率形式
            target_model_train = np.reshape(action_probs_, [1, self.action_num+1])

            k = reward + gamma * self.target_model.predict(target_model_train)  # 更新公式
            # print(k)
            y = np.reshape(k[0][0], [1, 1])
            self.Q_Table.loc[str(state), self.action_names[action_index]] = k[0][0]

            # 拟合训练
            history = self.model.fit(np.array(model_train), np.array(
                y), verbose=0)
            #
            loss = history.history['loss']
            sd.save_loss(self.name, loss)  # 保存本次loss值
            #
            self.target_model.set_weights(self.model.get_weights())  # 更新目标网络，主网络赋值给目标网络权重
            """调整tem"""
            self.current_t *= self.decay_rate
            print(self.name, '-LOSS:', loss)
            return loss
