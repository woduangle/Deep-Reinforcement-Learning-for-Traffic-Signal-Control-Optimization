import threading

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


class TLSAgent:
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
        self.nn_type = hyper_setting['nn_model']['type']
        self.input_dim = self.state_size
        self.input_shape = (self.state_size, 1)
        self.output_dim = self.action_size
        self.units = hyper_setting['nn_model']['units']
        self.active_func = hyper_setting['nn_model']['active_func']
        self.dropout = hyper_setting['nn_model']['dropout']
        self.loss_func = hyper_setting['nn_model']['loss_func']
        self.batch_size = hyper_setting['nn_model']['batch_size']
        self.epochs = hyper_setting['nn_model']['epochs']
        self.learning_rate = hyper_setting['nn_model']['learning_rate']
        #
        self.learning_model = hyper_setting['learning_model']['model']
        self.actor_learning_rate = hyper_setting['learning_model']['actor_learning_rate']
        self.critic_learning_rate = hyper_setting['learning_model']['critic_learning_rate']
        self.gamma = hyper_setting['learning_model']['gamma']
        self.maxlen = hyper_setting['learning_model']['maxlen']
        self.entropy_beta = hyper_setting['learning_model']['entropy_beta']
        """NN model"""
        self.actor_model, self.actor_target_model, self.critic_model, self.critic_target_model = self.neural_network(
            self.name)
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
        else:  # 根据当前状态选择最佳动作
            # 使用当前状态作为输入，通过神经网络模型预测各个动作的价值
            act_values = self.model_predict(self.state_current)
            # 从预测的动作价值中选择具有最高价值的动作作为当前动作
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
        sd.save_reward(self.name, val)  # 保存本次奖励值

    """
    以下为深度神经网络相关函数
    """

    def neural_network(self, name):
        """构造新的神经网络，或者读入保存的神经网络"""
        if os.path.exists(name + 'actor_model.h5'):
            # 如果已经存在名为 name + 'actor_model.h5' 的模型文件，则加载Actor模型和Critic模型
            actor_model = self.load_neural_networks(name=name, path='actor_model.h5')
            actor_target_model = self.load_neural_networks(
                name=name, path='actor_target model.h5')
            critic_model = self.load_neural_networks(name=name, path='critic_model.h5')
            critic_target_model = self.load_neural_networks(
                name=name, path='critic_target model.h5')

        else:  # 构造新的dnn
            # 否则，构建新的神经网络模型，构建Actor模型和Critic模型
            actor_model = self.build_nn(self.input_dim, self.output_dim)
            actor_target_model = self.build_nn(self.input_dim, self.output_dim)
            critic_model = self.build_nn(self.input_dim, 1)
            critic_target_model = self.build_nn(self.input_dim, 1)

        return actor_model, actor_target_model, critic_model, critic_target_model

    def model_predict(self, state):
        """模型预测"""
        # print(self.model.input_shape)
        state = np.expand_dims(state, axis=0)  # 扩维
        output = self.actor_model.predict(state)  # 使用Actor模型进行预测
        return output

    def load_neural_networks(self, name, path):
        """加载已保存的model"""
        return tf.keras.models.load_model(name + path)

    def build_nn(self, input_dim, output_dim):
        """构建多层感知器（MLP）模型"""
        model = self.build_mlp(input_dim, output_dim)
        return model

    def build_mlp(self, input_dim, output_dim):
        """创建MLP模型"""
        # 参数赋值
        lr = self.learning_rate  # 学习率
        units = self.units  # 隐藏层单元数
        active_func = self.active_func  # 激活函数
        loss_func = self.loss_func  # 损失函数
        """创建MLP模型"""
        model = Sequential()
        model.add(Dense(units=units, input_dim=input_dim, activation=active_func))  # 输入层
        model.add(Dense(units=32, activation='relu'))  # 隐藏层1
        model.add(Dense(units=16, activation='relu'))  # 隐藏层2
        # model.add(Dense(units=8, activation='relu'))
        model.add(Dense(units=output_dim, activation=active_func))  # 输出层
        model.summary()  # 输出模型结构信息
        model.compile(loss=loss_func, optimizer=Adam(learning_rate=lr))  # 编译模型
        return model

    def memorize(self, done=False):
        """保存<s,a,r,s'>到经验池"""
        self.reply_memory.append(
            (self.state_prev, self.action_current_index, self.reward_current, self.state_current, done))

    def experience_replay(self):
        """经验回放"""
        workers = []

        try:
            # 采样方法一：从经验池中随机抽样一批数据作为训练数据
            # 此方法在batch-size超过已有数量时，会产生异常
            mini_batch = random.sample(self.reply_memory, self.batch_size)
        except (ValueError):
            # 异常处理，当经验池中的样本数量不足以构成一个batch时，不进行之后的操作，直接返回
            return -1

        # 创建两个线程，分别用于训练 actor 和 critic
        t1 = threading.Thread(target=self.actor_train, args=(mini_batch,))  # 创建多线程
        t2 = threading.Thread(target=self.critic_train, args=(mini_batch,))  # 创建多线程
        workers.append(t1)
        workers.append(t2)

        # 启动线程
        for worker in workers:
            worker.start()

        # 等待所有线程结束
        for worker in workers:
            worker.join()

    def critic_train(self, mini_batch):
        """训练critic"""
        for state, action_index, reward, next_state, done in mini_batch:
            next_state = np.expand_dims(next_state, axis=0)  # 扩维，将下一个状态转换为神经网络接受的格式
            state = np.expand_dims(state, axis=0)  # 扩维，将当前状态转换为神经网络接受的格式
            next_v_value = self.critic_model.predict(next_state)  # 使用Critic模型预测下一个状态的价值
            td_targets = self.n_step_td_target(reward, next_v_value, done)  # 计算n步时间差分目标
            advantages = td_targets - self.critic_model.predict(state)  # 计算优势值

            with tf.GradientTape() as tape:  # tf.GradientTape() 是 TensorFlow 中用于计算梯度的上下文管理器
                v_pred = self.critic_model(state, training=True)  # 使用Critic模型预测当前状态的价值
                mse = tf.keras.losses.MeanSquaredError()  # 使用均方误差损失函数
                loss = mse(tf.stop_gradient(td_targets), v_pred)  # 计算损失
            grads = tape.gradient(loss, self.critic_model.trainable_variables)  # 计算梯度
            self.critic_model.optimizer.apply_gradients(zip(grads, self.critic_model.trainable_variables))  # 更新参数
            # self.save_neural_network(name, global_critic.model, critic.model)
            sd.save_loss(self.name + 'critic', float(loss))  # 保存本次loss值
            self.critic_target_model.set_weights(self.critic_model.get_weights())  # 更新目标模型的权重

    def actor_train(self, mini_batch):
        """训练actor"""
        for state, action_index, reward, next_state, done in mini_batch:
            next_state = np.expand_dims(next_state, axis=0)  # 扩维，将下一个状态转换为神经网络接受的格式
            state = np.expand_dims(state, axis=0)  # 扩维，将当前状态转换为神经网络接受的格式
            next_v_value = self.critic_model.predict(next_state)  # 使用Critic模型预测下一个状态的价值
            td_targets = self.n_step_td_target(reward, next_v_value, done)  # 计算n步时间差分目标
            advantages = td_targets - self.critic_model.predict(state)  # 计算优势值

            with tf.GradientTape() as tape:  # tf.GradientTape() 是 TensorFlow 中用于计算梯度的上下文管理器
                logits = self.actor_model(state, training=True)  # 获取Actor模型在当前状态下的输出概率分布

                ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # 定义交叉熵损失函数
                entropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)  # 定义熵损失函数
                actions = tf.cast(action_index, tf.int32)  # 将动作索引转换为张量
                policy_loss = ce_loss(actions, logits, sample_weight=tf.stop_gradient(advantages))  # 计算策略损失
                entropy = entropy_loss(logits, logits)  # 计算策略熵
                loss = policy_loss - self.entropy_beta * entropy  # 计算总损失

            grads = tape.gradient(loss, self.actor_model.trainable_variables)  # 计算梯度
            self.actor_model.optimizer.apply_gradients(zip(grads, self.actor_model.trainable_variables))  # 更新参数
            # self.actor_model.opt.apply_gradients(zip(grads, self.actor_model.trainable_variables))

            sd.save_loss(self.name + 'actor', float(loss))  # 保存本次loss值
            self.actor_target_model.set_weights(self.actor_model.get_weights())  # 更新目标模型的权重

            '''
            调整epsilon
            '''
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay  # 调整epsilon

            print(self.name, '-LOSS:', float(loss))  # 打印当前损失值

    def n_step_td_target(self, rewards, next_v_value, done):
        """计算 n-step TD 目标"""
        cumulative = 0  # 初始化累积奖励
        if not done:
            # 如果当前状态不是终止状态，则计算未来奖励的折扣累积
            cumulative = self.gamma * cumulative + rewards
            cumulative = np.reshape(cumulative, [1, 1])  # 将累积奖励调整为合适的形状
        else:
            # 如果当前状态是终止状态，则将下一个状态的估值作为累积奖励
            cumulative = next_v_value
        td_targets = cumulative  # 设置 TD 目标为累积奖励
        return td_targets
