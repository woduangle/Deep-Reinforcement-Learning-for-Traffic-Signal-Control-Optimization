from utils.process_control_nature_dqn_dnn import *


'''
读取Yaml文件数据
'''
yaml_data = read_yaml('yaml_config/_config_nature_dqn_dnn整个路网E3.yaml')
basic_setting = yaml_data[0]['BASIC_SETTING']  # 基本参数
hyper_setting = yaml_data[0]['HYPERPARAMETER_SETTING']  # 超参数
agent_setting = yaml_data[0]['AGENT_ELEMENT_SETTING']  # Agent参数