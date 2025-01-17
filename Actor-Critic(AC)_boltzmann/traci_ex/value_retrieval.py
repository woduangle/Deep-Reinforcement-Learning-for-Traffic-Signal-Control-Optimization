import traci


def adjust_number(x):
    """Only take the first two digits of this number, such as 2399, 2305, etc., become 2400"""
    if 0 < x < 1:
        return round(x, 2)  # If this number is less than 0, only retain the first two digits
    elif abs(x) < 100:
        return round(x)  # If this number is less than 100, round it down to the nearest whole number
    else:
        return adjust_number(x / 10) * 10


def adjust_decorator(func):
    """Define a decorator"""

    def wrapper(*args):
        val = func(*args)
        val = adjust_number(val)
        return val

    return wrapper


@adjust_decorator
def get_interval_occ_on_e2(e2_id):
    """Obtain the interval occupancy rate on the E2 detector"""
    val = traci.lanearea.getLastIntervalOccupancy(detID=e2_id)
    return val


@adjust_decorator
def get_interval_speed_on_e2(e2_id):
    """Obtain the interval average speed on the E2 detector"""
    val = traci.lanearea.getLastIntervalMeanSpeed(detID=e2_id)
    return val


@adjust_decorator
def get_interval_vol_on_e2(e2_id):
    """Obtain the interval vehicle count on the E2 detector"""
    val = traci.lanearea.getLastIntervalVehicleNumber(detID=e2_id)
    return val


@adjust_decorator
def get_interval_jam_length_on_e2(e2_id):
    """Obtain the interval congestion length on the E2 detector"""
    val = traci.lanearea.getLastIntervalMaxJamLengthInMeters(detID=e2_id)
    return val


@adjust_decorator
def get_occ_on_e2(e2_id):
    """Obtain the interval occupancy rate on the E2 detector (repeated, consider removing duplicate)"""
    val = traci.lanearea.getLastStepOccupancy(detID=e2_id)
    return val


@adjust_decorator
def get_speed_on_e2(e2_id):
    """Obtain the interval average speed on the E2 detector (repeated, consider removing duplicate)"""
    val = traci.lanearea.getLastStepMeanSpeed(detID=e2_id)
    return val


@adjust_decorator
def get_vol_on_e2(e2_id):
    """Obtain the interval vehicle count on the E2 detector (repeated, consider removing duplicate)"""
    val = traci.lanearea.getLastStepVehicleNumber(detID=e2_id)
    return val


@adjust_decorator
def get_jam_length_on_e2(e2_id):
    """Obtain the interval congestion length on the E2 detector (repeated, consider removing duplicate)"""
    val = traci.lanearea.getJamLengthMeters(detID=e2_id)
    return val


def get_interval_time_loss_on_e3_as_reward(e3_id):
    """The time loss on the E3 detector, as a reward, consider taking the reciprocal"""
    val = traci.multientryexit.getLastIntervalMeanTimeLoss(detID=e3_id)
    if val == -1 or val == 0:
        return 0
    else:
        return 1 / (val)
