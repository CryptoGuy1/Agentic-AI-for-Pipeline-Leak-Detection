def compute_reward(state, action):

    anomaly = state[0]

    reward = 0

    if anomaly < 0.5 and action == 0:
        reward += 1

    if anomaly > 0.8 and action == 3:
        reward += 2

    if anomaly > 0.9 and action == 4:
        reward += 3

    if anomaly > 0.8 and action == 0:
        reward -= 2

    return reward