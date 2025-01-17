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
from agent_ex.OU_noise import *
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
        self.actor_tau = hyper_setting['learning_model']['actor_tau']
        self.critic_learning_rate = hyper_setting['learning_model']['critic_learning_rate']
        self.critic_tau = hyper_setting['learning_model']['critic_tau']
        self.gamma = hyper_setting['learning_model']['gamma']
        self.maxlen = hyper_setting['learning_model']['maxlen']
        self.entropy_beta = hyper_setting['learning_model']['entropy_beta']
        """NN model"""
        self.actor_model, self.actor_target_model, self.critic_model, self.critic_target_model = self.neural_network(
            self.name)
        self.noise = OrnsteinUhlenbeckProcess(action_dimension=1)
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
        sd.save_reward(self.name, val)  # 保存本次奖励值

    """
    以下为深度神经网络相关函数
    """

    def neural_network(self, name):
        """构造新的神经网络，或者读入保存的神经网络"""
        if os.path.exists(name + 'actor_model.h5'):
            actor_model = self.load_neural_networks(name=name, path='actor_model.h5')
            actor_target_model = self.load_neural_networks(
                name=name, path='actor_target model.h5')
            critic_model = self.load_neural_networks(name=name, path='critic_model.h5')
            critic_target_model = self.load_neural_networks(
                name=name, path='critic_target model.h5')

        else:  # 构造新的dnn
            actor_model = self.build_nn(self.input_dim, self.output_dim)
            actor_target_model = self.build_nn(self.input_dim, self.output_dim)
            critic_model = self.build_nn(self.input_dim, 1)
            critic_target_model = self.build_nn(self.input_dim, 1)

        return actor_model, actor_target_model, critic_model, critic_target_model

    def model_predict(self, state):
        """模型预测"""
        # print(self.model.input_shape)
        # 噪化处理
        state_with_noise = OrnsteinUhlenbeckProcess.noise(self=self.noise, state=state)
        OrnsteinUhlenbeckProcess.reset(self=self.noise)

        state = state_with_noise
        state = np.expand_dims(state, axis=0)  # 扩维
        output = self.actor_model.predict(state)
        return output

    def load_neural_networks(self, name, path):
        """加载已保存的model"""
        return tf.keras.models.load_model(name + path)

    def build_nn(self, input_dim, output_dim):
        model = self.build_mlp(input_dim, output_dim)
        return model

    def build_mlp(self, input_dim, output_dim):
        """创建mlp"""
        # 参数赋值
        lr = self.learning_rate
        units = self.units
        active_func = self.active_func
        loss_func = self.loss_func
        """创建dnn"""
        model = Sequential()
        model.add(Dense(units=units, input_dim=input_dim, activation=active_func))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=16, activation='relu'))
        # model.add(Dense(units=8, activation='relu'))
        model.add(Dense(units=output_dim, activation=active_func))
        model.summary()
        model.compile(loss=loss_func, optimizer=Adam(learning_rate=lr))
        return model

    def memorize(self, done=False):
        """保存<s,a,r,s'>到经验池"""
        self.reply_memory.append(
            (self.state_prev, self.action_current_index, self.reward_current, self.state_current, done))



    def experience_replay(self):
        """经验回放"""
        workers = []

        try:
            # 采样方法一：此方法在batch-size超过已有数量时，会产生异常
            mini_batch = random.sample(self.reply_memory, self.batch_size)
        except (ValueError):
            # 异常处理，在数量不够时，不进行之后的操作
            return -1

        t1 = threading.Thread(target=self.actor_train, args=(mini_batch,))  # 创建多线程
        t2 = threading.Thread(target=self.critic_train, args=(mini_batch,))  # 创建多线程
        workers.append(t1)
        workers.append(t2)

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()

    def critic_train(self, mini_batch):
        """训练critic"""
        for state, action_index, reward, next_state, done in mini_batch:
            next_state = np.expand_dims(next_state, axis=0)  # 扩维
            state = np.expand_dims(state, axis=0)  # 扩维
            # next_state = np.reshape(next_state, [1, self.input_dim])
            # c_input = np.reshape(next_state, [1, self.input_dim])

            next_v_value = self.critic_model.predict(next_state)
            td_targets = self.n_step_td_target(reward, next_v_value, done)
            advantages = td_targets - self.critic_model.predict(state)

            with tf.GradientTape() as tape:
                v_pred = self.critic_model(state, training=True)
                mse = tf.keras.losses.MeanSquaredError()
                loss = mse(tf.stop_gradient(td_targets), v_pred)
            grads = tape.gradient(loss, self.critic_model.trainable_variables)
            self.critic_model.optimizer.apply_gradients(zip(grads, self.critic_model.trainable_variables))
            # self.save_neural_network(name, global_critic.model, critic.model)
            sd.save_loss(self.name + 'critic', float(loss))  # 保存本次loss值
            # 更新目标网络
            # self.critic_target_model.set_weights(self.critic_model.get_weights())
            critic_weights = self.critic_model.get_weights()
            critic_target_weights = self.critic_target_model.get_weights()
            critic_target_weights = [x * self.critic_tau + y * (1 - self.critic_tau) for x, y
                                     in zip(critic_target_weights, critic_weights)]
            self.critic_target_model.set_weights(critic_target_weights)

    def actor_train(self, mini_batch):
        """训练actor"""
        for state, action_index, reward, next_state, done in mini_batch:
            next_state = np.expand_dims(next_state, axis=0)  # 扩维
            state = np.expand_dims(state, axis=0)  # 扩维
            next_v_value = self.critic_model.predict(next_state)
            td_targets = self.n_step_td_target(reward, next_v_value, done)
            advantages = td_targets - self.critic_model.predict(state)

            with tf.GradientTape() as tape:
                logits = self.actor_model(state, training=True)

                ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                entropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
                actions = tf.cast(action_index, tf.int32)
                policy_loss = ce_loss(actions, logits, sample_weight=tf.stop_gradient(advantages))
                entropy = entropy_loss(logits, logits)
                loss = policy_loss - self.entropy_beta * entropy
            grads = tape.gradient(loss, self.actor_model.trainable_variables)
            self.actor_model.optimizer.apply_gradients(zip(grads, self.actor_model.trainable_variables))
            # self.actor_model.opt.apply_gradients(zip(grads, self.actor_model.trainable_variables))

            sd.save_loss(self.name + 'actor', float(loss))  # 保存本次loss值
            # 更新目标网络
            # self.actor_target_model.set_weights(self.actor_model.get_weights())
            actor_weights = self.actor_model.get_weights()
            actor_target_weights = self.actor_target_model.get_weights()
            actor_target_weights = [x * self.actor_tau + y * (1 - self.actor_tau) for x, y
                                    in zip(actor_target_weights, actor_weights)]
            self.actor_target_model.set_weights(actor_target_weights)

            '''
            调整epsilon
            '''
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            print(self.name, '-LOSS:', float(loss))

    def n_step_td_target(self, rewards, next_v_value, done):
        cumulative = 0
        if not done:
            cumulative = self.gamma * cumulative + rewards
            cumulative = np.reshape(cumulative, [1, 1])
        else:
            cumulative = next_v_value
        td_targets = cumulative
        return td_targets
