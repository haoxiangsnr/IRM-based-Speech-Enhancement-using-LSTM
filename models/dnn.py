import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self, n_input_frames=7, n_fft=257, n_hidden_units=2048):
        super().__init__()
        self.main = nn.Sequential(
            # Layer 1
            nn.Linear(n_input_frames * n_fft, n_hidden_units),
            # https://www.zhihu.com/search?type=content&q=batch%20normalization
            # Batch Norm 与 activate function 的使用顺序
            nn.BatchNorm1d(n_hidden_units),
            nn.ReLU(),

            # Layer 2
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.BatchNorm1d(n_hidden_units),
            nn.ReLU(),

            # Layer 3
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.BatchNorm1d(n_hidden_units),
            nn.ReLU(),

            # layer 4
            nn.Linear(n_hidden_units, n_fft),
            nn.BatchNorm1d(n_fft),
            nn.Sigmoid()
        )

    def forward(self, ipt):
        return self.main(ipt)
