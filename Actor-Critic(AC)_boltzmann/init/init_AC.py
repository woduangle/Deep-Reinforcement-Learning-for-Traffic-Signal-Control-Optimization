import sys
import os
import time
import save_data.save_data as sd
import shutil

os.chdir(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))  # Set the working directory
from utils.process_control_AC_boltzmann import *

'''
Read Yaml file data
'''
yaml_data = read_yaml('yaml_config/_config_Actor-Critic(AC).yaml')
basic_setting = yaml_data[0]['BASIC_SETTING']  # Basic parameters
rl_setting = yaml_data[0]['PARAMETER_SETTING']  # Hyperparameters
agent_setting = yaml_data[0]['AGENT_ELEMENT_SETTING']  # Agent parameters
