import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from model import FinReportModel
from data_loader import load_data
from preprocessing import select_features, normalize_features
from tqdm import tqdm  # For progress tracking

# Define a simple dataset for sequence data
class FinDataset(Dataset):
    def __init__(self, features, labels, seq_len):
        self.features = features
        self.labels = labels
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.features) - self.seq_len)

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.seq_len]
        y = self.labels[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)

# Function to create DataLoaders given training and validation indices
def create_dataloaders(features, labels, seq_len, batch_size, train_idx, val_idx):
    train_features = features[train_idx]
    train_labels = labels[train_idx]
    val_features = features[val_idx]
    val_labels = labels[val_idx]
    
    train_dataset = FinDataset(train_features, train_labels, seq_len)
    val_dataset = FinDataset(val_features, val_labels, seq_len)
    
    # Shuffle training data for better optimization
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# Training and evaluation function that returns average validation loss
def train_and_evaluate(train_loader, val_loader, input_size, hidden_size, num_layers, dropout, learning_rate, num_epochs):
    model = FinReportModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # Optionally, print epoch progress:
        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss, model

def main():
    # Load configuration from config.yaml (for default values and paths)
    with open('src/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    data_path = config['data_path']
    default_batch_size = config['batch_size']
    
    # Load the full dataset
    df = load_data(data_path)
    
    # Use select_features to extract only numeric features and labels
    features, labels = select_features(df)
    features, scaler = normalize_features(features)
    
    features = np.array(features)
    labels = np.array(labels)
    
    # Extend hyperparameter grids:
    learning_rates = [0.001, 0.0005, 0.0001]
    hidden_sizes = [64, 128, 256]
    num_layers_options = [1, 2, 3]
    seq_lengths = [5, 10, 15]
    dropouts = [0.0, 0.2, 0.4]
    num_epochs = config['num_epochs']
    batch_size = default_batch_size
    
    best_val_loss = float('inf')
    best_params = None
    results = []
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Grid search over hyperparameters using tqdm for progress
    for lr in tqdm(learning_rates, desc="Learning Rates"):
        for hidden in hidden_sizes:
            for layers in num_layers_options:
                for seq_len in seq_lengths:
                    for dropout in dropouts:
                        val_losses = []
                        for train_idx, val_idx in tscv.split(features):
                            if len(train_idx) <= seq_len or len(val_idx) <= seq_len:
                                continue
                            
                            train_loader, val_loader = create_dataloaders(features, labels, seq_len, batch_size, train_idx, val_idx)
                            avg_val_loss, _ = train_and_evaluate(
                                train_loader, val_loader,
                                input_size=features.shape[1],
                                hidden_size=hidden,
                                num_layers=layers,
                                dropout=dropout,
                                learning_rate=lr,
                                num_epochs=num_epochs
                            )
                            val_losses.append(avg_val_loss)
                        if val_losses:
                            avg_loss = np.mean(val_losses)
                            print(f"lr: {lr}, hidden: {hidden}, layers: {layers}, seq_len: {seq_len}, dropout: {dropout}, avg_val_loss: {avg_loss:.4f}")
                            results.append({
                                'learning_rate': lr,
                                'hidden_size': hidden,
                                'num_layers': layers,
                                'seq_len': seq_len,
                                'dropout': dropout,
                                'avg_val_loss': avg_loss
                            })
                            if avg_loss < best_val_loss:
                                best_val_loss = avg_loss
                                best_params = {
                                    'learning_rate': lr,
                                    'hidden_size': hidden,
                                    'num_layers': layers,
                                    'seq_len': seq_len,
                                    'dropout': dropout
                                }
                        else:
                            print(f"Skipped hyperparameter set: lr={lr}, hidden={hidden}, layers={layers}, seq_len={seq_len}, dropout={dropout} due to insufficient data.")
    
    print("\nBest Hyperparameters Found:")
    print(best_params)
    print("Best Validation Loss:", best_val_loss)
    
    import pandas as pd
    results_df = pd.DataFrame(results)
    print("\nGrid Search Results:")
    print(results_df.sort_values(by='avg_val_loss'))
    
if __name__ == "__main__":
    main()
