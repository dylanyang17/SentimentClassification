import torch
from torch import nn
from config import Config


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, Config.filter_num, kernel_size=(3, Config.vec_len)),
            nn.ReLU(),
            nn.MaxPool2d((Config.seq_len - 3 + 1, 1)),
            nn.Dropout(p=.2),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(1, Config.filter_num, kernel_size=(4, Config.vec_len)),
            nn.ReLU(),
            nn.MaxPool2d((Config.seq_len - 4 + 1, 1)),
            nn.Dropout(p=.2),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(1, Config.filter_num, kernel_size=(5, Config.vec_len)),
            nn.ReLU(),
            nn.MaxPool2d((Config.seq_len - 5 + 1, 1)),
            nn.Dropout(p=.2),
        )
        self.fc = nn.Linear(3 * Config.filter_num, Config.label_len)

    def forward(self, input_data):
        x1 = self.cnn1(input_data.unsqueeze(1))
        x2 = self.cnn2(input_data.unsqueeze(1))
        x3 = self.cnn3(input_data.unsqueeze(1))
        output = torch.cat((x1, x2, x3), -1)
        output = output.view(input_data.shape[0], 1, -1)
        output = self.fc(output)
        output = output.view(-1, Config.label_len)
        return output
