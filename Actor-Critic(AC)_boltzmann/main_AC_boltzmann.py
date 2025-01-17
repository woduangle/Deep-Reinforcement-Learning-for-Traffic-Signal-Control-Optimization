from init.init_AC import *


if __name__ == '__main__':
    '''
    Initialize the Agent and learning algorithm parameters
    '''
    tls_agents = initialize_agents_by(agent_setting,rl_setting)  #
    '''
        Begin the training episode
    '''
    for ep in range(basic_setting['episode']):
        print('-------------Training episode in progress: ', ep, '-------------')
        start_env(sumo_cmd=basic_setting['sumo_start_config_cmd'])  # Initialize the environment
        prerun_env(pre_steps=basic_setting['pre_steps'])  # Preheat the environment
        # Start running the simulation environment
        start_time = get_env_time()  # The moment the simulation environment begins
        while get_env_time() < start_time + basic_setting['simulation_time']:

            for agent in agent_setting.keys():  # raverse all Agents
                # Obtain the current action execution status of the Agent, whether it is completed
                if tls_agents[agent].is_current_strategy_over() == False:  # The current action is being executed...
                    continue  # If the current agent's policy has not been fully executed, take no action
                # Obtain the current stat
                # get_current_state(agent=tls_agents[agent])
                # Select an action
                select_action(agent=tls_agents[agent])
                # Write the action into the environment
                deploy_action_into_env(agent=tls_agents[agent])
            # Execute one step of simulation
            simulation_step()
            # Decrement the strategy execution timer for all Agents by 1
            time_count_step(agent_list=tls_agents)
            # raverse all Agents
            for agent in agent_setting.keys():
                # Re-obtain the action execution status of the Agent, check if it is complete
                if tls_agents[agent].is_current_strategy_over() == False:  # The current action is being executed...
                    continue  # If the current agent's policy has not been fully executed, take no action
                # Obtain the reward
                get_reward(agent=tls_agents[agent])
                # Obtain the current state
                get_current_state(agent=tls_agents[agent])
                # Update the Q-table
                update_q_table(agent=tls_agents[agent])

        # Close
        print("The current episode ends：", ep)
        '''Close the simulation'''
        close_env()

        '''Save the data for this episode'''
        sd.save_ep_data(ep=ep)

        '''Save the SUMO output data'''
        save_sumo_output_data(ep)

    '''
    The run ends, save the data to a file
    '''
    sd.save_data_to_file()

