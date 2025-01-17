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



class DDQNDNN_TLSAgent:
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
        self.state_config = agent_setting['states']
        self.action_config = agent_setting['actions']  # action config
        self.reward_config = agent_setting['rewards']  # reward config
        #
        """初始化基本参数"""
        self.state_names = self.state_config['names']
        self.action_names = self.action_config['names']
        self.state_num = len(self.state_names)  # 状态数量
        self.state_size = np.size(self.state_names)  # 状态空间大小
        self.action_num = len(self.action_names)
        self.action_size = np.size(self.action_names)
        self.action_duration = [i[2] for _, i in self.action_config['paras'].items()]  # 动作持续时间
        # 初始值
        self.state_prev = [0] * self.state_num  # 得到全为0的空的动作空间
        self.state_current = [0] * self.state_num
        self.action_current = random.choice(self.action_names)
        self.action_current_index = self.action_names.index(self.action_current)
        self.reward_current = 0.0
        """记录当前Agent的动作类型，及当前动作的剩余时间"""
        self.current_strategy_remain_time = 0  # 当前策略剩余时间，为0时表示需要重新指定策略
        """动作选择：参数"""
        self.action_selection = hyper_setting['action_selection']['model']
        self.epsilon = hyper_setting['action_selection']['epsilon']
        self.epsilon_min = hyper_setting['action_selection']['epsilon_min']
        self.epsilon_decay = hyper_setting['action_selection']['epsilon_decay']
        """深度学习超参数"""
        self.dnn_type = hyper_setting['dnn_model']['type']
        self.input_dim = self.state_size
        self.input_shape = (self.state_size, 1)
        self.units = hyper_setting['dnn_model']['units']
        self.active_func = hyper_setting['dnn_model']['active_func']
        self.dropout = hyper_setting['dnn_model']['dropout']
        self.loss_func = hyper_setting['dnn_model']['loss_func']
        self.batch_size = hyper_setting['dnn_model']['batch_size']
        self.epochs = hyper_setting['dnn_model']['epochs']
        #
        self.learning_model = hyper_setting['learning_model']['model']
        self.learning_rate = hyper_setting['learning_model']['learning_rate']
        self.gamma = hyper_setting['learning_model']['gamma']
        self.maxlen = hyper_setting['learning_model']['maxlen']
        """DNN model"""
        self.model, self.target_model = self.neural_network(self.name)
        """replay memory"""
        self.reply_memory = deque(maxlen=self.maxlen)  # 双向队列

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
            action = np.random.choice(self.action_num)
        else:
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

    """
    以下为深度神经网络相关函数
    """

    def neural_network(self, name):
        """构造新的神经网络，或者读入保存的神经网络"""
        if os.path.exists(name + 'model.h5'):  # 检查是否存在已保存的模型文件
            model = self.load_neural_networks(name=name, path='model.h5')  # 加载已保存的神经网络模型
            target_model = self.load_neural_networks(name=name, path='target model.h5')  # 加载目标网络模型

        else:  # 如果没有已保存的模型文件，则构造新的神经网络
            model = self.build_dnn()  # 构建新的神经网络
            target_model = self.build_dnn()  # 构建新的目标网络

        return model, target_model

    def model_predict(self, state):
        """模型预测"""
        # print(self.model.input_shape)
        state = np.expand_dims(state, axis=0)  # 扩维，将状态扩展为神经网络接受的格式
        output = self.model.predict(state)  # 使用模型预测动作值
        return output

    def load_neural_networks(self, name, path):
        """加载已保存的model"""
        return tf.keras.models.load_model(name + path)  # 加载指定路径下的模型

    def build_dnn(self):
        """创建dnn"""
        # 设置神经网络的参数
        lr = self.learning_rate  # 学习率
        input_dim = self.input_dim  # 输入维度
        output_dim = self.action_size  # 输出维度（动作空间大小）
        units = self.units  # 神经网络隐藏层单元数
        active_func = self.active_func  # 激活函数
        loss_func = self.loss_func  # 损失函数

        """创建dnn"""
        # 构建神经网络模型
        model = Sequential()  # 创建Sequential模型
        model.add(Dense(units=units, input_dim=input_dim, activation=active_func))  # 添加输入层和隐藏层
        model.add(Dense(units=32, activation='relu'))  # 添加隐藏层
        model.add(Dense(units=16, activation='relu'))  # 添加隐藏层
        model.add(Dense(units=output_dim, activation=active_func))  # 添加输出层
        model.summary()  # 打印模型结构信息
        model.compile(loss=loss_func, optimizer=Adam(learning_rate=lr))  # 编译模型，指定损失函数和优化器
        return model  # 返回构建好的神经网络模型

    def memorize(self, done=False):
        """保存<s,a,r,s'>到经验池"""
        # 将状态、动作、奖励和下一个状态存储到经验池中
        self.reply_memory.append(
            (self.state_prev, self.action_current_index, self.reward_current, self.state_current, done))

    def experience_replay(self):
        """经验回放"""
        gamma = self.gamma  # 设置折扣因子
        epochs = self.epochs  # 设置迭代轮数

        try:
            # 采样方法一：此方法在batch-size超过已有数量时，会产生异常
            mini_batch = random.sample(self.reply_memory, self.batch_size)  # 从经验池中随机采样一个批次的数据
        except (ValueError):
            # 异常处理，在数量不够时，不进行之后的操作
            return -1

        state_f, target_f = [], []  # for training， 用于存储训练数据的列表

        for state, action_index, reward, next_state, done in mini_batch:
            next_state = np.expand_dims(next_state, axis=0)  # 将下一个状态扩展为神经网络接受的格式
            state = np.expand_dims(state, axis=0)  # 将当前状态扩展为神经网络接受的格式

            target = 0
            if done:
                target = reward  # 如果当前状态是终止状态，目标值等于即时奖励
            else:
                # 否则，目标值为即时奖励加上折扣因子乘以下一状态的最大动作值的估计值
                target = reward + gamma * np.max(self.target_model.predict(next_state)[0])  # 计算目标值

            target_t = self.model.predict(state)  # 使用当前模型预测当前状态的动作值
            target_t[0][action_index] = target  # 更新目标值
            # filtering out states and targets for training， 将状态和目标值添加到训练数据列表中
            state_f.append(state[0])
            target_f.append(target_t[0])

        # 使用训练数据进行模型训练
        history = self.model.fit(np.array(state_f), np.array(
            target_f), epochs=epochs, verbose=0)
        #
        loss = history.history['loss'][0]  # 获取训练损失
        sd.save_loss(self.name, loss)  # 保存本次loss值
        #
        self.target_model.set_weights(self.model.get_weights())  # 更新目标网络的权重

        '''
        调整epsilon
        '''
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        print(self.name, '-LOSS:', loss)
        return loss
