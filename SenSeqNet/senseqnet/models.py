# senseqnet/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout_rate):
        """
        input_dim = 1280
        hidden_dim = 181
        num_layers = 4
        num_classes = 2
        dropout_rate = 0.4397133138964481
        """
        super(ImprovedLSTMClassifier, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        # Kept for checkpoint compatibility; not applied in forward.
        self.bn = nn.BatchNorm1d(hidden_dim * 2)

        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=76, kernel_size=(6, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(76)

        self.conv2 = nn.Conv2d(in_channels=76, out_channels=111, kernel_size=(4, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(111)

        self.conv3 = nn.Conv2d(in_channels=111, out_channels=487, kernel_size=(5, 1), padding=(1, 0))
        self.bn3 = nn.BatchNorm2d(487)

        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        # Dynamically compute flatten dimension with a dummy batch
        self.flatten_dim = self._get_flatten_dim(input_dim, hidden_dim, num_layers)

        self.dropout = nn.Dropout(0.5456158649892608)
        self.fc = nn.Linear(self.flatten_dim, num_classes)

    def _get_flatten_dim(self, input_dim, hidden_dim, num_layers):
        """
        Pass a dummy batch of size 32 through the network to discover the
        final flattened dimension for the fully-connected layer.
        Adjust the dummy batch size if your actual batch_size differs.
        """
        batch_size = 32  # same dummy size you used in training
        # Create dummy input shape: (batch_size=32, seq_len=1, input_dim=1280)
        dummy_x = torch.ones(batch_size, 1, input_dim)
        
        # Dummy initial hidden & cell states
        h0 = torch.zeros(num_layers * 2, batch_size, hidden_dim)
        c0 = torch.zeros(num_layers * 2, batch_size, hidden_dim)

        # LSTM forward
        out, _ = self.lstm(dummy_x, (h0, c0))
        # Take the last time step: (batch_size, hidden_dim*2)
        out = out[:, -1, :]

        # CNN path
        out = out.unsqueeze(1).unsqueeze(-1)  # (batch_size, 1, hidden_dim*2, 1)
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.pool(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.pool(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)
        out = self.pool(out)

        # Flatten
        out = out.view(out.size(0), -1)
        return out.size(1)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len=1, 1280)
        """
        batch_sz = x.size(0)

        # Initialize hidden and cell states
        h0 = torch.zeros(
            self.lstm.num_layers * 2,
            batch_sz,
            self.lstm.hidden_size,
            device=x.device
        )
        c0 = torch.zeros(
            self.lstm.num_layers * 2,
            batch_sz,
            self.lstm.hidden_size,
            device=x.device
        )

        # LSTM forward
        out, _ = self.lstm(x, (h0, c0))
        # Take last time step
        out = out[:, -1, :]  # (batch_sz, hidden_dim * 2)

        # CNN path
        out = out.unsqueeze(1).unsqueeze(-1)  # (batch_sz, 1, hidden_dim*2, 1)
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.pool(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.pool(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)
        out = self.pool(out)

        # Flatten
        out = out.view(out.size(0), -1)

        # Dropout + FC
        out = self.dropout(out)
        out = self.fc(out)

        return F.log_softmax(out, dim=1)
