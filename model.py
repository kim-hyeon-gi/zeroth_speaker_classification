import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.autograd import Variable


def model_list():
    output_size = 115
    p = 0.2
    batch_size = 64
    return [
        conv1D_Model(output_size, p),
        conv2D_Model(output_size, p),
        transformer_Model(output_size, p),
        mean_Model(output_size, p),
        lstm_Model(output_size, p, batch_size=batch_size),
    ]


class conv1D_Model(nn.Module):
    def __init__(self, output_size, p):
        super().__init__()

        self.layers2 = nn.Sequential(
            nn.Conv1d(in_channels=40, out_channels=30, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=5, stride=2),
        )

        self.layers4 = nn.Sequential(
            nn.Linear(3660, 1600),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1600),
            nn.Dropout(p),
            nn.Linear(1600, 800),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(800),
            nn.Dropout(p),
            nn.Linear(800, 400),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(400),
            nn.Dropout(p),
            nn.Linear(400, output_size),
            nn.Softmax(),
        )

    def forward(self, x):

        x = self.layers2(x)

        x = torch.flatten(x, start_dim=1)
        x = self.layers4(x)
        return x


class conv2D_Model(nn.Module):
    def __init__(self, output_size, p):
        super().__init__()

        self.layers2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=2),
        )

        self.layers4 = nn.Sequential(
            nn.Linear(8540, 4000),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4000),
            nn.Dropout(p),
            nn.Linear(4000, 2000),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2000),
            nn.Dropout(p),
            nn.Linear(2000, output_size),
            nn.Softmax(),
        )

    def forward(self, x):
        x = x.view(1, 64, 40, 500)
        print(np.shape(x))
        x = self.layers2(x)

        x = torch.flatten(x, start_dim=1)
        x = self.layers4(x)
        return x


class transformer_Model(nn.Module):
    def __init__(self, output_size, p):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=40, nhead=8)

        self.layers2 = nn.Sequential(nn.TransformerEncoder(encoder_layer, num_layers=6))

        self.layers4 = nn.Sequential(
            nn.Linear(40, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p),
            nn.Linear(256, output_size),
            nn.Softmax(),
        )

    def forward(self, x):
        x = self.layers2(x)
        x = x.mean(dim=1)
        x = self.layers4(x)
        return x


class mean_Model(nn.Module):
    def __init__(self, output_size, p):
        super().__init__()

        self.layers1 = nn.Sequential(
            nn.Linear(40, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p),
            nn.Linear(512, 300),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(300),
            nn.Dropout(p),
            nn.Linear(300, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p),
            nn.Linear(256, output_size),
            nn.Softmax(),
        )

    def forward(self, x):

        x = x.mean(dim=2)
        x = self.layers1(x)
        return x


class lstm_Model(nn.Module):
    def __init__(self, output_size, p, batch_size):
        super(lstm_Model, self).__init__()
        self.hidden_size = 128
        self.batch_size = batch_size
        self.num_layers = 5
        self.lstm = nn.LSTM(
            input_size=40,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.3,
        )

        self.layers4 = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p),
            nn.Linear(512, output_size),
            nn.Softmax(),
        )

    def forward(self, X):
        h_0 = Variable(
            torch.zeros(self.num_layers, self.batch_size, self.hidden_size).cuda()
        )
        c_0 = Variable(
            torch.zeros(self.num_layers, self.batch_size, self.hidden_size).cuda()
        )
        output, hidden = self.lstm(X, (h_0, c_0))

        output = output[:, -1]  # 최종 예측 Hidden Layer

        output = self.layers4(output)
        return output
