from traci_ex.basic_commands import *
from traci_ex.value_retrieval import *
from traci_ex.value_changing import *
# import traci_ex
import yaml
from agent_ex.ddqn_dnn_tls_agent import *
# import agent_ex
import shutil


def read_yaml(yaml_file):
    """to read a yaml file"""
    with open(yaml_file, 'rb') as f:
        all_data = list(yaml.safe_load_all(f))
    return all_data


def start_env(sumo_cmd):
    """启动仿真"""
    print('初始化仿真环境……')
    simulation_start(sumoCmd=sumo_cmd)


def close_env():
    """关闭环境"""
    simulation_close()


def prerun_env(pre_steps):
    """仿真预热"""
    pre_run_simulation_to_prepare(pre_steps=pre_steps)


def get_env_time():
    """获取当前仿真的时间"""
    return get_simulation_time()


def env_step():
    """运行环境前进一步"""
    simulation_step()


def initialize_agents_by(*args):
    """Initialize all intersection agents according to agent settings"""
    agent_settings = args[0]
    hyper_setting = args[1]
    intersection_agents = {}
    for key, agent_setting in agent_settings.items():
        intersection_agents[key] = DDQNDNN_TLSAgent(
            agent_setting, hyper_setting)
    return intersection_agents


def get_current_state(agent):
    """获得Agent当前状态"""
    state_config = agent.state_config
    state_val = []  # 初始化状态
    for state_name in state_config['names']:  # 遍历每一个状态变量
        func_name = state_config['func_names'][state_name]
        paras = state_config['paras'][state_name]
        val = eval(func_name)(paras)
        state_val.append(val)
    agent.save_current_state(state_val)


def select_action(agent):
    """选择动作"""
    agent.select_action()


def deploy_action_into_env(agent):
    """将动作部署到环境中，等待执行"""
    action_config = agent.action_config
    action_name = agent.action_current
    print('action_current:', action_name)
    func_name = action_config['func_names'][action_name]
    paras = action_config['paras'][action_name]
    # 运行函数
    eval(func_name)(paras)


def time_count_step(agent_list):
    """将所有Agent的策略执行时间计数器-1，相当于执行1 step"""
    for agent in agent_list:
        agent_list[agent].time_count_step()


def get_reward(agent):
    """获取奖励"""
    reward_config = agent.reward_config
    reward_name = reward_config['names']  # 奖励只有一个
    func_name = reward_config['func_names'][reward_name]
    paras = reward_config['paras'][reward_name]
    #
    val = eval(func_name)(paras)
    agent.save_reward(val)


def memorize(agent, done=False):
    """保存<s,a,r,s'>到经验池"""
    agent.memorize(done=done)


def experience_replay(agent_list):
    """经验回放"""
    for agent in agent_list:
        agent_list[agent].experience_replay()

def save_sumo_output_data(ep):
    """保存SUMO仿真输出数据"""
    sumo_dir = os.path.join(os.getcwd(), 'sumo_network') # sumo仿真文件夹
    default_output_dir = os.path.join(sumo_dir, 'output') # 仿真输出数据，默认文件夹
    dir_name = f"output_data({ep})" # 输出数据文件夹：output_data(99)
    new_output_dir = os.path.join(sumo_dir, dir_name) # 含路径

    if os.path.exists(new_output_dir) == True: #如果存在，则清空所有文件
        shutil.rmtree(new_output_dir) # 删除文件及文件夹
    else:
        pass
    #
    shutil.copytree(default_output_dir, new_output_dir)
