import traci


def simulation_start(sumoCmd, port=8813, label="sim1"):
    """初始化仿真traci"""
    traci.start(cmd=sumoCmd, port=port, label=label)
    print(traci.getVersion())


def simulation_step():
    """ 执行一步仿真"""
    traci.simulationStep()


def simulation_close():
    """关闭仿真"""
    traci.close(wait=False)


def pre_run_simulation_to_prepare(pre_steps):
    """仿真预热准备"""
    traci.simulationStep(traci.simulation.getTime() + pre_steps)


def get_simulation_time():
    """获取当前仿真的时间"""
    return traci.simulation.getTime()
