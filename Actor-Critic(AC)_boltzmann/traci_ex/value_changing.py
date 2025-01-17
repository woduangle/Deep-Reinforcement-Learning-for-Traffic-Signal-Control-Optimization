import traci

def switch_to_tls_program(*args):
    """Switch to the specified control scheme"""
    tls_id = args[0][0]
    p_id = args[0][1]
    traci.trafficlight.setProgram(tlsID=tls_id,programID=p_id)

def multi_junctions_switch_to_tls_program(*args):
    """For multiple intersections, switch control schemes simultaneously"""
    tls_ids = args[0][0]
    p_id = args[0][1]
    for tls_id in tls_ids:
        traci.trafficlight.setProgram(tlsID=tls_id,programID=p_id)