from torch import nn
from config import Config


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(Config.seq_len, Config.seq_len, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(Config.seq_len),
            nn.Dropout(p=.2),

            nn.Conv1d(Config.seq_len, Config.seq_len, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(Config.seq_len),
            nn.Dropout(p=.2),

            nn.Conv1d(Config.seq_len, 16, kernel_size=5, padding=2),
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
