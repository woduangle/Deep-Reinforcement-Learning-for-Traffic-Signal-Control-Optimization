from init.init_NSHG_DQN import *
import net_game.net_game as ng

if __name__ == '__main__':
    '''
    初始化Agent及学习算法相关参数、收益矩阵
    '''
    tls_agents = initialize_agents_by(agent_setting, hyper_setting)  #
    ng.init_player_list(agent_setting)
    '''
        开始训练回合
    '''
    for ep in range(basic_setting['episode']):
        print('-------------训练回合: ', ep, '-------------')
        start_env(sumo_cmd=basic_setting['sumo_start_config_cmd'])  # 初始化环境
        prerun_env(pre_steps=basic_setting['pre_steps'])  # 环境预热
        # 开始运行仿真环境
        start_time = get_env_time()  # 仿真环境开始的时刻
        while get_env_time() < start_time + basic_setting['simulation_time']:

            for agent in agent_setting.keys():  # 遍历所有Agent
                # 获取Agent当前动作执行情况，是否执行完成
                if tls_agents[agent].is_current_strategy_over() == False:  # 当前动作执行中。。。
                    continue  # 如果当前agent的策略还没有执行完，不做操作
                # 获取状态
                # get_current_state(agent=tls_agents[agent])
                # 选择动作
                select_action(agent=tls_agents[agent])
                # 将动作写入环境中
                deploy_action_into_env(agent=tls_agents[agent])
            # 执行一步仿真
            simulation_step()
            # 将所有Agent策略执行时间计数器-1
            time_count_step(agent_list=tls_agents)
            # 遍历所有Agent
            for agent in agent_setting.keys():
                # 重新获取Agent的动作执行情况，是否完成
                if tls_agents[agent].is_current_strategy_over() == False:  # 当前动作执行中。。。
                    continue  # 如果当前agent的策略还没有执行完，不做操作
                # 获取奖励
                get_reward(agent=tls_agents[agent])
                # 获取状态
                get_current_state(agent=tls_agents[agent])
                # 保存<s,a,r,s'>到经验池
                memorize(agent_id=agent, agents=tls_agents, done=False)


        # 关闭
        print("本次epsido结束：", ep)
        '''关闭仿真'''
        close_env()

        '''
        经验回放
        '''
        experience_replay(agent_list=tls_agents)

        '''保存本次episode数据'''
        sd.save_ep_data(ep=ep)

        '''保存SUMO输出数据'''
        save_sumo_output_data(ep)

    '''
    运行结束，将数据保存到文件
    '''
    sd.save_data_to_file()

