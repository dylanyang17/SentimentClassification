import torch
from torch import nn
from config import Config


class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=Config.vec_len, hidden_size=5,
                            num_layers=1, dropout=0.2)
        self.fc = nn.Linear(5, Config.label_len)

    def forward(self, input_data):
        states, hidden = self.lstm(input_data.permute([1, 0, 2]))
        outputs = self.fc(states[-1])
        return outputs
