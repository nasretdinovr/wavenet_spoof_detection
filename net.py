import torch
import torchvision.models as models
from kymatio import Scattering1D
from kymatio.datasets import fetch_fsdd
import torch.nn as nn


class CNNNet(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNNet, self).__init__()
        self.J = 6
        self.Q = 8
        self.T = 40000
        self.num_classes = num_classes
        self.output_size = 125
        self.hidden_size_1 = 2048
        self.hidden_size_2 = 1024
        self.hidden_size_3 = 512
        self.hidden_size_4 = 256
        self.hidden_size_5 = 128
        self.hidden_size_6 = 64

        self.log_eps = 1e-6

        self.scattering = Scattering1D(self.J, self.T, self.Q)
        self.scattering.cuda()
        self.h1 = nn.Linear(self.output_size, self.hidden_size_1)
        self.h2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.h3 = nn.Linear(self.hidden_size_2, self.hidden_size_3)
        self.h4 = nn.Linear(self.hidden_size_3, self.hidden_size_4)
        self.h5 = nn.Linear(self.hidden_size_4, self.hidden_size_5)
        self.h6 = nn.Linear(self.hidden_size_5, self.hidden_size_6)
        self.output = nn.Linear(self.hidden_size_6, self.num_classes)

        self.relu = nn.ReLU()
        self.do = nn.Dropout()

    def forward(self, data):
        x = self.scattering.forward(data)
        x = x[:, 1:, :]
        x = torch.log(torch.abs(x) + self.log_eps)
        x = torch.mean(x, dim=-1)
        print(torch.mean(x, dim=-1))
        x = self.h1(x)
        x = self.relu(x)
        # x = self.do(x)
        x = self.h2(x)
        x = self.relu(x)
        x = self.h3(x)
        x = self.relu(x)
        x = self.h4(x)
        x = self.relu(x)
        x = self.h5(x)
        x = self.relu(x)
        x = self.h6(x)
        x = self.relu(x)
        return self.output(x)