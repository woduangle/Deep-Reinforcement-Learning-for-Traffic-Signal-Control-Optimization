from init.import_modules import *
from init.init_dnn import *

# import init


def main():
    '''
    初始化Agent
    '''
    tls_agents = initialize_agents_by(agent_setting, hyper_setting)

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
                # 获取新状态
                get_current_state(agent=tls_agents[agent])
                # 获取奖励
                get_reward(agent=tls_agents[agent])
                # 保存<s,a,r,s'>到经验池
                memorize(agent=tls_agents[agent], done=False)
                # 选择动作
                select_action(agent=tls_agents[agent])  # 为什么一开始不进行动作选择？ 因为此处的动作选择是下一轮的
                # 将动作写入环境中
                deploy_action_into_env(agent=tls_agents[agent])
            # 运行环境执行一个步长
            env_step()
            # 将所有Agent策略执行时间计数器-1
            time_count_step(agent_list=tls_agents)
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


class TimerContext:
    """程序运行计时器"""

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time()
        print(f"Time taken: {self.end - self.start}秒")


if __name__ == "__main__":
    with TimerContext():
        main()
