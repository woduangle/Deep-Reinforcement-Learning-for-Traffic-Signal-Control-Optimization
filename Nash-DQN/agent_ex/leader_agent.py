import pandas as pd
import numpy as np
import os
import random
from collections import deque
import tensorflow as tf
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout, SimpleRNN, LSTM, GRU, Input
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical, plot_model
import save_data.save_data as sd  # 保存数据
import net_game.net_game as ng


class Leader_Agent:
    """
    交叉口信号控制器Agent类
    """

    def __init__(self, *args):
        agent_setting = args[0]  # 状态、动作、奖励设施  AGENT_ELEMENT_SETTING
        hyper_setting = args[1]  # 超参数  HYPERPARAMETER_SETTING
        """Initialization"""
        print('Agent初始化……')
        self.type = 'TLS'
        # agent_id denoted as cross id
        self.name = agent_setting['id']
        self.agent_id = agent_setting['id']
        self.cross_id = self.agent_id
        self.tls_id = agent_setting['id']  # 信号灯id
        # basic info about states of the agent
        self.state_config = agent_setting['states']  # 状态配置
        self.action_config = agent_setting['actions']  # action config
        self.reward_config = agent_setting['rewards']  # reward config
        #
        """初始化基本参数"""
        self.state_names = self.state_config['names']  # 状态名称列表
        self.action_names = self.action_config['names']  # 动作名称列表
        self.state_num = len(self.state_names)  # 状态数量
        self.state_size = np.size(self.state_names)  # 状态空间大小
        self.action_num = len(self.action_names)
        self.action_size = np.size(self.action_names)
        self.action_duration = [i[2] for _, i in self.action_config['paras'].items()]  # 动作持续时间
        # 初始值
        self.state_prev = [0] * self.state_num  # 上一个状态（初始为全 0）得到全为0的空的状态空间
        self.state_current = [0] * self.state_num
        self.action_current = random.choice(self.action_names)  # 当前动作（初始随机选择）
        self.action_current_index = self.action_names.index(self.action_current)  # 当前动作索引
        self.reward_current = 0.0  # 当前奖励
        self.exp_action_list = []  # 博弈专用经验动作列表
        """记录当前Agent的动作类型，及当前动作的剩余时间"""
        self.current_strategy_remain_time = 0  # 当前策略剩余时间，为0时表示需要重新指定策略
        """动作选择：参数"""
        self.action_selection = hyper_setting['action_selection']['model']
        self.epsilon = hyper_setting['action_selection']['epsilon']
        self.epsilon_min = hyper_setting['action_selection']['epsilon_min']
        self.epsilon_decay = hyper_setting['action_selection']['epsilon_decay']
        """网络博弈"""
        self.identity = agent_setting['identity']
        self.opponent = agent_setting['leader_opponent']
        """深度学习超参数"""
        self.dnn_type = hyper_setting['dnn_model']['type']  # 模型类型
        self.input_dim = self.state_size  # 输入维度
        self.input_shape = (self.state_size, 1)  # 输入形状
        self.units = hyper_setting['dnn_model']['units']  # 隐藏层单元数
        self.active_func = hyper_setting['dnn_model']['active_func']  # 激活函数
        self.dropout = hyper_setting['dnn_model']['dropout']  # dropout
        self.loss_func = hyper_setting['dnn_model']['loss_func']  # 损失函数
        self.batch_size = hyper_setting['dnn_model']['batch_size']  # 批量大小
        self.epochs = hyper_setting['dnn_model']['epochs']  # 训练轮数
        #
        self.learning_model = hyper_setting['learning_model']['model']
        self.learning_rate = hyper_setting['learning_model']['learning_rate']
        self.gamma = hyper_setting['learning_model']['gamma']
        self.maxlen = hyper_setting['learning_model']['maxlen']
        """DNN model"""
        self.model, self.target_model = self.neural_network(self.name)  # 构建深度神经网络模型
        """replay memory"""
        self.reply_memory = deque(maxlen=self.maxlen)  # 初始化回放内存，双向队列

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
        """根据当前状态选择动作，epsilon greedy"""
        if np.random.rand() <= self.epsilon:
            # 以 ε 的概率随机选择动作
            action = np.random.choice(self.action_num)
        else:
            # 否则，根据当前状态选择最佳动作
            act_values = self.model_predict(self.state_current)
            action = np.argmax(act_values[0])
        # 保存动作，并设置计时器
        self.save_action_set_time_counter(action_index=action)

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
        print(self.name, '-REWARD:', val)
        self.reward_current = val
        sd.save_reward(self.name,val)  # 保存本次奖励值

    def save_exp_action_list(self, action_list):
        """保存博弈专用经验动作列表"""
        self.exp_action_list = action_list

    """
    以下为深度神经网络相关函数
    """

    def neural_network(self, name):
        """构造新的神经网络，或者读入保存的神经网络"""
        if os.path.exists(name + 'model.h5'):
            # 如果已经存在名为 name + 'model.h5' 的模型文件，则加载模型
            model = self.load_neural_networks(name=name, path='model.h5')
            target_model = self.load_neural_networks(
                name=name, path='target model.h5')

        else:  # 构造新的dnn
            # 如果不存在，则构造新的神经网络
            model = self.build_dnn()
            target_model = self.build_dnn()

        return model, target_model

    def model_predict(self, state):
        """模型预测"""
        # print(self.model.input_shape)
        state = np.expand_dims(state, axis=0)  # 扩维，将输入状态转换为神经网络接受的格式
        output = self.model.predict(state)  # 使用神经网络预测输出
        return output

    def load_neural_networks(self, name, path):
        """加载已保存的model"""
        return tf.keras.models.load_model(name + path)

    def build_dnn(self):
        """创建dnn"""
        # 参数赋值
        lr = self.learning_rate  # 学习率
        input_dim = self.input_dim  # 输入维度
        output_dim = self.action_size  # 输出维度
        units = self.units  # 隐藏层单元数
        active_func = self.active_func  # 激活函数
        loss_func = self.loss_func  # 损失函数
        """创建dnn"""
        model = Sequential()  # 创建Sequential模型
        # 添加输入层
        model.add(Dense(units=units, input_dim=input_dim, activation=active_func))
        # 添加两个隐藏层
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=16, activation='relu'))
        # 添加输出层
        model.add(Dense(units=output_dim, activation=active_func))
        model.summary()  # 输出模型结构信息
        model.compile(loss=loss_func, optimizer=Adam(learning_rate=lr))   # 编译模型
        return model

    def memorize(self, done=False):
        """保存<s,a,r,s'>到经验池"""
        self.reply_memory.append(
            (self.state_prev, self.action_current_index, self.reward_current, self.state_current, self.exp_action_list, done))

    def experience_replay(self):
        """经验回放"""
        gamma = self.gamma  # 折扣因子
        epochs = self.epochs  # 训练轮数

        try:
            # 采样方法一：此方法在batch-size超过已有数量时，会产生异常
            mini_batch = random.sample(self.reply_memory, self.batch_size)
        except (ValueError):
            # 异常处理，在数量不够时，不进行之后的操作
            return -1

        state_f, target_f = [], []  # for training

        for state, action_index, reward, next_state, action_list, done in mini_batch:
            next_state = np.expand_dims(next_state, axis=0)  # 扩维，将下一个状态转换为神经网络接受的格式
            state = np.expand_dims(state, axis=0)  # 扩维，将当前状态转换为神经网络接受的格式

            target = 0

            """calculate nash value of agents"""
            nash_values_dic = ng.get_leaders_NE_values_by()  # 计算得到纳什均衡值
            print(nash_values_dic)
            """give game value"""
            nash_q_value = nash_values_dic[self.name]  # 获取当前智能体的纳什均衡值

            if done:
                target = reward  # 如果已经结束，目标值为即时奖励
            else:
                target = reward + gamma * nash_q_value  # 如果未结束，计算目标值，目标值为即时奖励加上纳什均衡值

            target_t = self.model.predict(state)  # 使用神经网络预测当前状态的输出值
            target_t[0][action_index] = target  # 更新目标值
            # filtering out states and targets for training
            state_f.append(state[0])  # 添加当前状态到训练数据中
            target_f.append(target_t[0])  # 添加目标值到训练数据中
            #
            ng.update_leader_NE_payoff_table(self.name, payoff=target, my_action=action_list[0],
                                             your_action=action_list[1])  # 更新纳什均衡收益表

        # 训练模型
        history = self.model.fit(np.array(state_f), np.array(
            target_f), epochs=epochs, verbose=0)
        #
        loss = history.history['loss'][0]  # 获取损失值
        sd.save_loss(self.name, loss)  # 保存本次loss值
        #
        self.target_model.set_weights(self.model.get_weights())  # 更新目标模型的权重



        '''
        调整epsilon
        '''
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # 调整ε值

        print(self.name, '-LOSS:', loss)
        return loss
