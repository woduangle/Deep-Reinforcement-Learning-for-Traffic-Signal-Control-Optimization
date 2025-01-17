import traci

def adjust_number(x):
    """只取这个数字的前两位，2399、2305等变2400"""
    if abs(x)<100:
        return round(x)
    else:
        return adjust_number(x/10) * 10
    
def adjust_decorator(func):
    """定义decorator"""
    def wrapper(*args):
        val = func(*args)
        val = adjust_number(val)
        return val
    return wrapper

@adjust_decorator
def get_interval_occ_on_e2(e2_id):
    """获取E2检测器上的interval 占有率"""
    val = traci.lanearea.getLastIntervalOccupancy(detID=e2_id)
    return val
    
@adjust_decorator
def get_interval_speed_on_e2(e2_id):
    """获取E2检测器上的interval 平均速度"""
    val = traci.lanearea.getLastIntervalMeanSpeed(detID=e2_id)
    return val
    
@adjust_decorator
def get_interval_vol_on_e2(e2_id):
    """获取E2检测器上的interval 车辆数"""
    val = traci.lanearea.getLastIntervalVehicleNumber(detID=e2_id)
    return val
    
@adjust_decorator
def get_interval_jam_length_on_e2(e2_id):
    """获取E2检测器上的interval 拥堵长度"""
    val = traci.lanearea.getLastIntervalMaxJamLengthInMeters(detID=e2_id)
    return val

@adjust_decorator
def get_occ_on_e2(e2_id):
    """获取E2检测器上的interval 占有率"""
    val = traci.lanearea.getLastStepOccupancy(detID=e2_id)
    return val

@adjust_decorator
def get_speed_on_e2(e2_id):
    """获取E2检测器上的interval 平均速度"""
    val = traci.lanearea.getLastStepMeanSpeed(detID=e2_id)
    return val

@adjust_decorator
def get_vol_on_e2(e2_id):
    """获取E2检测器上的interval 车辆数"""
    val = traci.lanearea.getLastStepVehicleNumber(detID=e2_id)
    return val

@adjust_decorator
def get_jam_length_on_e2(e2_id):
    """获取E2检测器上的interval 拥堵长度"""
    val = traci.lanearea.getJamLengthMeters(detID=e2_id)
    return val


def get_interval_time_loss_on_e3_as_reward(e3_id):
    """E3检测器上的time loss，作为奖励，考虑倒数"""
    val = traci.multientryexit.getLastIntervalMeanTimeLoss(detID=e3_id)
    if val == -1 or val == 0:
        return 0
    else:
        return 1/(val)

