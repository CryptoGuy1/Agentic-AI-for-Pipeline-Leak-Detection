import torch
import torch.nn as nn
import torch.nn.functional as F

# Define your DQN network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DecisionTool:

    def __init__(self, model_path, state_dim=5, action_dim=4):
        print("Loading DQN model...")
        self.model = DQN(state_dim, action_dim)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def decide(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(state_tensor)

        action = torch.argmax(q_values).item()

        return action, q_values.tolist()[0]