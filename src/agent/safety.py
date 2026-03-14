
def safety_override(state, action):
    
    anomaly = state[0]

    if anomaly > 0.95:
        return 4

    return action