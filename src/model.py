import torch
import torch.nn as nn
import torch.nn.functional as F

class FinReportModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        """
        LSTM model for financial report analysis with regularization.
        
        Args:
            input_size (int): Size of input features
            hidden_size (int): Size of hidden layers
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate (0-1)
        """
        super(FinReportModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Add batch normalization (keep this for training stability)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)  # Will be 0.0 based on optimal hyperparameters
        self.fc = nn.Linear(hidden_size, 1)
        
        # Initialize weights more safely
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights with Xavier/Glorot initialization for tensors with 2+ dimensions only"""
        for name, param in self.named_parameters():
            if 'weight' in name and 'lstm' not in name:
                # Only apply Xavier initialization to tensors with dimension >= 2
                if len(param.shape) >= 2:
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.normal_(param, mean=0.0, std=0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        """Forward pass with regularization"""
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        
        # Get the last time step output
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply batch normalization
        normalized = self.batch_norm(last_hidden)
        
        # Apply dropout for regularization
        dropped = self.dropout(normalized)
        
        # Final linear layer
        out = self.fc(dropped)
        
        return out.squeeze()  # (batch_size,)
    
    def predict_with_uncertainty(self, x, mc_samples=10):
        """
        Perform Monte Carlo dropout prediction to estimate uncertainty
        
        Args:
            x: Input tensor
            mc_samples: Number of Monte Carlo samples
            
        Returns:
            tuple: (mean prediction, prediction std dev)
        """
        self.train()  # Set to train mode to enable dropout
        
        predictions = []
        for _ in range(mc_samples):
            with torch.no_grad():
                output = self.forward(x)
                predictions.append(output.unsqueeze(0))
                
        predictions = torch.cat(predictions, dim=0)
        mean_pred = torch.mean(predictions, dim=0)
        std_pred = torch.std(predictions, dim=0)
        
        return mean_pred, std_pred