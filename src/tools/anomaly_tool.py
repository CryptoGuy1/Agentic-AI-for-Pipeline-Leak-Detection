import numpy as np
import torch

class AnomalyTool:
    def __init__(self, model_path):
        self.model = torch.load(model_path, map_location="cpu")
        self.model.eval()

    def normalize(self, data):
        # Manual normalization (safe for Pi)
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)

        # Prevent divide-by-zero
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        return normalized

    def compute(self, sensor_data):
        sensor_data = self.normalize(sensor_data)

        tensor = torch.tensor(sensor_data, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            reconstructed = self.model(tensor)
            loss = torch.mean((tensor - reconstructed) ** 2)

        return loss.item()