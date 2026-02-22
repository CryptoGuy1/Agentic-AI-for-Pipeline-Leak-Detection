def safety_override(state, action):
    anomaly, gas_class, conf, temp, hum = state

    # Example rule:
    # If anomaly is extremely high, force emergency shutdown
    if anomaly > 0.95:
        return 3  # emergency_shutdown

    return action