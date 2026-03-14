import torch
import torch.nn as nn


# -----------------------------
# RL Decision Network
# -----------------------------
class DecisionNetwork(nn.Module):

    def __init__(self, state_dim=5, action_dim=5):

        super().__init__()

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# -----------------------------
# Decision Tool
# -----------------------------
class DecisionTool:

    def __init__(self, model_path):

        # create model
        self.model = DecisionNetwork()

        # load weights
        weights = torch.load(model_path, map_location="cpu")

        self.model.load_state_dict(weights)

        self.model.eval()

        self.actions = {
            0: "monitor",
            1: "increase_sampling",
            2: "request_verification",
            3: "raise_alert",
            4: "emergency_shutdown"
        }

    def decide(self, state):

        x = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(x)

        action = torch.argmax(q_values).item()

        return action, q_values.numpy()[0]