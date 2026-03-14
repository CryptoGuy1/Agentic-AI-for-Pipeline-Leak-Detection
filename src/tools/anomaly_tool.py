import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):

    def __init__(self, input_size=7, hidden_size=32):

        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):

        encoded, _ = self.encoder(x)

        decoded, _ = self.decoder(encoded)

        out = self.output_layer(decoded)

        return out


class AnomalyTool:

    def __init__(self, model_path):

        # build model
        self.model = LSTMAutoencoder()

        # load weights
        weights = torch.load(model_path, map_location="cpu")

        self.model.load_state_dict(weights)

        self.model.eval()

    def compute(self, sensor_array):

        x = torch.FloatTensor(sensor_array).unsqueeze(0)

        with torch.no_grad():

            reconstruction = self.model(x)

        loss = torch.mean((x - reconstruction) ** 2).item()

        return loss