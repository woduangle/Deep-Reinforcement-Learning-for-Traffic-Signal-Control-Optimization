import sys
import os
import time
import save_data.save_data as sd
import shutil

from utils.process_control_NSHG_DQN import *
os.chdir(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))  # 设置工作目录

'''
读取Yaml文件数据
'''
yaml_data = read_yaml('yaml_config/_config_NSHG_DQN.yaml')
basic_setting = yaml_data[0]['BASIC_SETTING']  # 基本参数
hyper_setting = yaml_data[0]['HYPERPARAMETER_SETTING']  # 超参数
agent_setting = yaml_data[0]['AGENT_ELEMENT_SETTING']  # Agent参数
