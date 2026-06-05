import torch
import torch.nn as nn


### 샘플모델 수정예정
import torch
import torch.nn as nn

class LSTM(nn.Module):

    def __init__(
        self,
        input_dim=12,
        hidden_dim=64,
        num_layers=2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):

        # x:
        # (batch, seq_len, feature)

        out, _ = self.lstm(x)

        # out:
        # (batch, seq_len, hidden_dim)

        out = self.fc(out)

        # (batch, seq_len, 1)

        return out


class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)