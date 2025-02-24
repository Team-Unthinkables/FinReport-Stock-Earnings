import torch.nn as nn

class FinReportModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(FinReportModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return out.squeeze()
