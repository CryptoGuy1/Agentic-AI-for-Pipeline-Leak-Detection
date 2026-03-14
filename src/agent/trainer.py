class AgentTrainer:

    def __init__(self, decision_model):

        self.model = decision_model

    def train(self, memory):

        batch = memory.get_recent()

        for item in batch:

            state = item["state"]
            reward = item["reward"]

            # update Q network
            self.model.update(state, reward)