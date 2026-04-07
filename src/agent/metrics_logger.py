class MetricsLogger:

    def __init__(self):
        self.episode_rewards = []
        self.current_episode_reward = 0

        self.actions = []
        self.true_labels = []

        self.episode_numbers = []

    def log_step(self, action, reward, true_label):
        self.actions.append(action)
        self.true_labels.append(true_label)
        self.current_episode_reward += reward

    def end_episode(self, episode):
        self.episode_rewards.append(self.current_episode_reward)
        self.episode_numbers.append(episode)
        self.current_episode_reward = 0