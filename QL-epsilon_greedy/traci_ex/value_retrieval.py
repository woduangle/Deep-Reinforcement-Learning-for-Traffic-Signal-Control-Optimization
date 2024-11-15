import traci


def adjust_number(x):
    """only take the first two digits of this number, 2399, 2305, etc. become 2400"""
    if 0 < x < 1:
        return round(x, 2)  # if this number is less than 0, only the first two digits are retained
    elif abs(x) < 100:
        return round(x)  # if this number is less than 100, take the integer
    else:
        return adjust_number(x / 10) * 10


def adjust_decorator(func):
    """define decorator"""

    def wrapper(*args):
        val = func(*args)
        val = adjust_number(val)
        return val

    return wrapper


@adjust_decorator
def get_interval_occ_on_e2(e2_id):
    """obtain interval occupancy rate on E2 detector"""
    val = traci.lanearea.getLastIntervalOccupancy(detID=e2_id)
    return val


@adjust_decorator
def get_interval_speed_on_e2(e2_id):
    """obtain the interval average velocity on the E2 detector"""
    val = traci.lanearea.getLastIntervalMeanSpeed(detID=e2_id)
    return val


@adjust_decorator
def get_interval_vol_on_e2(e2_id):
    """obtain the number of interval vehicles on the E2 detector"""
    val = traci.lanearea.getLastIntervalVehicleNumber(detID=e2_id)
    return val


@adjust_decorator
def get_interval_jam_length_on_e2(e2_id):
    """obtain the interval congestion length on the E2 detector"""
    val = traci.lanearea.getLastIntervalMaxJamLengthInMeters(detID=e2_id)
    return val


@adjust_decorator
def get_occ_on_e2(e2_id):
    """obtain interval occupancy rate on E2 detector"""
    val = traci.lanearea.getLastStepOccupancy(detID=e2_id)
    return val


@adjust_decorator
def get_speed_on_e2(e2_id):
    """obtain the interval average velocity on the E2 detector"""
    val = traci.lanearea.getLastStepMeanSpeed(detID=e2_id)
    return val


@adjust_decorator
def get_vol_on_e2(e2_id):
    """obtain the number of interval vehicles on the E2 detector"""
    val = traci.lanearea.getLastStepVehicleNumber(detID=e2_id)
    return val


@adjust_decorator
def get_jam_length_on_e2(e2_id):
    """obtain the interval congestion length on the E2 detector"""
    val = traci.lanearea.getJamLengthMeters(detID=e2_id)
    return val


def get_interval_time_loss_on_e3_as_reward(e3_id):
    """time loss on E3 detector, as a reward, consider the reciprocal"""
    val = traci.multientryexit.getLastIntervalMeanTimeLoss(detID=e3_id)
    if val == -1 or val == 0:
        return 0
    else:
        return 1 / (val)
