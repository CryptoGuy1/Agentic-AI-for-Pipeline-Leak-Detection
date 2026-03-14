class SupervisorAgent:

    def __init__(self, worker_agent):

        self.worker = worker_agent

    def run_cycle(self, image, sensors, temp, hum, step):

        result = self.worker.run_once(
            image=image,
            sensor_array=sensors,
            temp=temp,
            hum=hum,
            step=step
        )

        # Supervisor oversight
        anomaly = result["state"][0]
        action = result["action"]

        # Safety escalation
        if anomaly > 0.7 and action < 3:

            print("SUPERVISOR OVERRIDE: forcing alarm")

            result["action"] = 3

        return result