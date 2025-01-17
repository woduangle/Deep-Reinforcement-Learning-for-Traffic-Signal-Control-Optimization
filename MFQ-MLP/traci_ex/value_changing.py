import traci

def switch_to_tls_program(*args):
    """切换到指定的控制方案"""
    tls_id = args[0][0]
    p_id = args[0][1]
    traci.trafficlight.setProgram(tlsID=tls_id,programID=p_id)

def multi_junctions_switch_to_tls_program(*args):
    """针对多个路口，同时切换控制方案"""
    tls_ids = args[0][0]
    p_id = args[0][1]
    for tls_id in tls_ids:
        traci.trafficlight.setProgram(tlsID=tls_id,programID=p_id)