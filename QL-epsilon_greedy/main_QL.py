from init.init_QL import *


if __name__ == '__main__':
    '''
    initialize Agent and learn algorithm related parameters
    '''
    tls_agents = initialize_agents_by(agent_setting,rl_setting)  #
    '''
        start training round
    '''
    for ep in range(basic_setting['episode']):
        print('-------------training round: ', ep, '-------------')
        start_env(sumo_cmd=basic_setting['sumo_start_config_cmd'])  # initialize environment
        prerun_env(pre_steps=basic_setting['pre_steps'])  # environmental preheating
        # start running the simulation environment
        start_time = get_env_time()  # the moment when the simulation environment starts
        while get_env_time() < start_time + basic_setting['simulation_time']:

            for agent in agent_setting.keys():  # traverse all agents
                # obtain the current execution status of the Agent's actions and whether they have been completed
                if tls_agents[agent].is_current_strategy_over() == False:  # the current action is being executed。。。
                    continue  # if the current agent's strategy has not been fully executed, no action will be taken
                # get status
                # get_current_state(agent=tls_agents[agent])
                # select action
                select_action(agent=tls_agents[agent])
                # write actions into the environment
                deploy_action_into_env(agent=tls_agents[agent])
            # perform a one-step simulation
            simulation_step()
            # subtract all Agent policy execution time counters by one
            time_count_step(agent_list=tls_agents)
            # traverse all agents
            for agent in agent_setting.keys():
                # retrieve the execution status of the Agent's actions and confirm if it has been completed
                if tls_agents[agent].is_current_strategy_over() == False:  # the current action is being executed。。。
                    continue  # if the current agent's strategy has not been fully executed, no action will be taken
                # get reward
                get_reward(agent=tls_agents[agent])
                # get state
                get_current_state(agent=tls_agents[agent])
                # update table
                update_q_table(agent=tls_agents[agent])

        # close
        print("this epsido has ended：", ep)
        '''ciose simulation'''
        close_env()

        '''save the data for this episode'''
        sd.save_ep_data(ep=ep)

        '''save SUMO output data'''
        save_sumo_output_data(ep)

    '''
    run completed, save data to file
    '''
    sd.save_data_to_file()

