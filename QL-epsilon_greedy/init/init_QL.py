import sys
import os
import time
import save_data.save_data as sd
import shutil

os.chdir(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))  # set up work directory
from utils.process_control_QL_epsilon_greedy import *

'''
read Yaml file data
'''
yaml_data = read_yaml('yaml_config/_config_QL-epsilon.yaml')
basic_setting = yaml_data[0]['BASIC_SETTING']  # basic parameters
rl_setting = yaml_data[0]['PARAMETER_SETTING']  # hyper-parameters
agent_setting = yaml_data[0]['AGENT_ELEMENT_SETTING']  # Agent parameters
