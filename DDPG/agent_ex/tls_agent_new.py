import threading
import pandas as pd
import numpy as np
import os
import random
from collections import deque
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Sequential, Model
from keras.optimizers import Adam
import save_data.save_data as sd  # 保存数据
from agent_ex.OU_noise import OrnsteinUhlenbeckProcess


class TLSAgent:
    """
    交叉口信号控制器Agent类
    """

    def __init__(self, agent_setting, hyper_setting):
        """
        Agent初始化方法
        """
        print('Agent初始化……')
        self.type = 'TLS'
        self.name = agent_setting['id']
        self.cross_id = self.agent_id = self.tls_id = agent_setting['id']
        self.state_config = agent_setting['states']
        self.action_config = agent_setting['actions']
        self.reward_config = agent_setting['rewards']
        # 状态相关参数
        self.state_names = self.state_config['names']
        self.state_num = len(self.state_names)
        self.state_size = np.size(self.state_names)
        self.state_prev = [0] * self.state_num
        self.state_current = [0] * self.state_num
        # 动作相关参数
        self.action_names = self.action_config['names']
        self.action_num = len(self.action_names)
        self.action_size = np.size(self.action_names)
        self.action_duration = [i[2] for _, i in self.action_config['paras'].items()]
        self.action_current = random.choice(self.action_names)
        self.action_current_index = self.action_names.index(self.action_current)
        # 初始化奖励
        self.reward_current = 0.0
        # 策略剩余时间
        self.current_strategy_remain_time = 0
        # 动作选择参数
        self.epsilon = hyper_setting['action_selection']['epsilon']
        self.epsilon_min = hyper_setting['action_selection']['epsilon_min']
        self.epsilon_decay = hyper_setting['action_selection']['epsilon_decay']
        # 深度学习超参数
        self.learning_rate = hyper_setting['nn_model']['learning_rate']
        self.batch_size = hyper_setting['nn_model']['batch_size']
        self.epochs = hyper_setting['nn_model']['epochs']
        # NN模型
        self.actor_model, self.actor_target_model, self.critic_model, self.critic_target_model = self.neural_network()
        self.noise = OrnsteinUhlenbeckProcess(action_dimension=1)
        # 经验回放记忆
        self.reply_memory = deque(maxlen=hyper_setting['learning_model']['maxlen'])

    def is_current_strategy_over(self):
        """判断当前动作是否执行完成"""
        return self.current_strategy_remain_time == 0

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

    def save_reward(self, val):
        """保存当前的奖励"""
        print(self.name, '-REWARD:', val)
        self.reward_current = val
        sd.save_reward(self.name, val)  # 保存本次奖励值

    """
    以下为深度神经网络相关函数
    """

    def neural_network(self):
        """构造新的神经网络，或者读入保存的神经网络"""
        if os.path.exists(self.name + 'actor_model.h5'):
            actor_model = self.load_neural_networks(name=self.name, path='actor_model.h5')
            actor_target_model = self.load_neural_networks(name=self.name, path='actor_target model.h5')
            critic_model = self.load_neural_networks(name=self.name, path='critic_model.h5')
            critic_target_model = self.load_neural_networks(name=self.name, path='critic_target model.h5')
        else:
            actor_model = self.build_nn()
            actor_target_model = self.build_nn()
            critic_model = self.build_nn()
            critic_target_model = self.build_nn()
        return actor_model, actor_target_model, critic_model, critic_target_model

    def model_predict(self, state):
        """模型预测"""
        state_with_noise = OrnsteinUhlenbeckProcess.noise(self=self.noise, state=state)
        OrnsteinUhlenbeckProcess.reset(self=self.noise)
        state = state_with_noise
        state = np.expand_dims(state, axis=0)
        output = self.actor_model.predict(state)
        return output

    def load_neural_networks(self, name, path):
        """加载已保存的model"""
        return tf.keras.models.load_model(name + path)

    def build_nn(self):
        """创建神经网络"""
        model = Sequential()
        model.add(Dense(units=32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(units=16, activation='relu'))
        model.add(Dense(units=self.action_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, done=False):
        """保存<s,a,r,s'>到经验池"""
        self.reply_memory.append(
            (self.state_prev, self.action_current_index, self.reward_current, self.state_current, done))

    def experience_replay(self):
        """经验回放"""
        mini_batch = random.sample(self.reply_memory, min(len(self.reply_memory), self.batch_size))
        t1 = threading.Thread(target=self.actor_train, args=(mini_batch,))
        t2 = threading.Thread(target=self.critic_train, args=(mini_batch,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    def critic_train(self, mini_batch):
        """训练critic"""
        for state, action_index, reward, next_state, done in mini_batch:
            next_state = np.expand_dims(next_state, axis=0)
            state = np.expand_dims(state, axis=0)
            next_v_value = self.critic_model.predict(next_state)
            td_targets = self.n_step_td_target(reward, next_v_value, done)
            advantages = td_targets - self.critic_model.predict(state)
            with tf.GradientTape() as tape:
                v_pred = self.critic_model(state, training=True)
                mse = tf.keras.losses.MeanSquaredError()
                loss = mse(tf.stop_gradient(td_targets), v_pred)
            grads = tape.gradient(loss, self.critic_model.trainable_variables)
            self.critic_model.optimizer.apply_gradients(zip(grads, self.critic_model.trainable_variables))

    def actor_train(self, mini_batch):
        """训练actor"""
        for state, action_index, reward, next_state, done in mini_batch:
            next_state = np.expand_dims(next_state, axis=0)
            state = np.expand_dims(state, axis=0)
            next_v_value = self.critic_model.predict(next_state)
            td_targets = self.n_step_td_target(reward, next_v_value, done)
            advantages = td_targets - self.critic_model.predict(state)
            with tf.GradientTape() as tape:
                logits = self.actor_model(state, training=True)
                ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                policy_loss = ce_loss(tf.cast(action_index, tf.int32), logits,
                                      sample_weight=tf.stop_gradient(advantages))
                loss = policy_loss
            grads = tape.gradient(loss, self.actor_model.trainable_variables)
            self.actor_model.optimizer.apply_gradients(zip(grads, self.actor_model.trainable_variables))

    def n_step_td_target(self, rewards, next_v_value, done):
        cumulative = 0
        if not done:
            cumulative = self.gamma * cumulative + rewards
            cumulative = np.reshape(cumulative, [1, 1])
        else:
            cumulative = next_v_value
        td_targets = cumulative
        return td_targets
