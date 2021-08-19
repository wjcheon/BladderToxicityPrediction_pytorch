import torch
import torch.nn as nn
import torch.nn.functional as F



class FCN(nn.Module):
    def __init__(self, input_ch=18, output_ch=5):
        super(FCN, self).__init__()
        self.layer_1 = nn.Linear(input_ch, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_4 = nn.Linear(64, 32)
        self.layer_5 = nn.Linear(32, 16)
        self.layer_out = nn.Linear(16, output_ch)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)
        self.batchnorm4 = nn.BatchNorm1d(32)
        self.batchnorm5 = nn.BatchNorm1d(16)

    def forward(self, x):
        # x = torch.flatten(x)

        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_4(x)
        x = self.batchnorm4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_5(x)
        x = self.batchnorm5(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)


        # x1 = self.fc1(x)
        # x1 = self.dropout(x1)
        # x1 = self.LeakyReLU(x1)
        #
        # x2 = self.fc2(x1)
        # x2 = self.dropout(x2)
        # x2 = self.LeakyReLU(x2)
        #
        # x3 = self.fc3(x2)
        # x3 = self.dropout(x3)
        # x3 = self.LeakyReLU(x3)
        #
        # x4 = self.fc4(x3)
        # x4 = self.dropout(x4)
        # x4 = self.LeakyReLU(x4)
        #
        # x5 = self.fc5(x4)
        # x5 = self.dropout(x5)
        # x5 = self.LeakyReLU(x5)
        #
        # x6 = self.fc6(x5)
        # #out = F.log_softmax(x6, dim=1)
        # out = x6


        return x

