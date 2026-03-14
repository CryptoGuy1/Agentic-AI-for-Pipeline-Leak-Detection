def encode_state(anomaly, gas_class, conf, temp, hum):

    gas_map = {
        "nogas": 0,
        "methane": 1,
        "propane": 2,
        "butane": 3
    }

    gas_numeric = gas_map.get(gas_class, 0)

    state = [
        anomaly,
        gas_numeric,
        conf,
        temp / 100,
        hum / 100
    ]

    return state