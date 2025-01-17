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
            # 使用当前网络选择下一个状态的最佳动作
            next_action_index = np.argmax(self.model_predict(next_state)[0])
            # 使用目标网络评估选择的动作的 Q 值
            target_q_value = self.target_model.predict(next_state)[0][next_action_index]
            target = reward + gamma * target_q_value  # 如果未结束，计算目标值，目标值为即时奖励加上纳什均衡值

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
