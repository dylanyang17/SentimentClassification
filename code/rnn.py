from torch import nn
from config import Config


class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.Sequential(
            nn.LSTM(Config.seq_len, Config.seq_len, 2),
            nn.Conv1d(Config.seq_len, 200, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Dropout(p=.2),

            nn.Conv1d(200, 100, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Dropout(p=.2),

            nn.Conv1d(100, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(p=.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * Config.vec_len, 100),
            nn.ReLU(),
            nn.Linear(100, 8),
            nn.Tanh()
        )

    def forward(self, input_data):
        output = self.cnn(input_data)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output
