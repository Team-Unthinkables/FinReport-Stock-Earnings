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
from tqdm import tqdm

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
def create_dataloaders(features, labels, seq_len, batch_size, train_idx, val_idx, num_workers=2):
    train_features = features[train_idx]
    train_labels = labels[train_idx]
    val_features = features[val_idx]
    val_labels = labels[val_idx]
    
    train_dataset = FinDataset(train_features, train_labels, seq_len)
    val_dataset = FinDataset(val_features, val_labels, seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader

# Training and evaluation function with early stopping
def train_and_evaluate(train_loader, val_loader, input_size, hidden_size, num_layers, dropout, learning_rate, num_epochs, device, patience=5):
    model = FinReportModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
        
        # Evaluate on validation set at the end of each epoch
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Uncomment the next line to see epoch progress:
        # print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {avg_val_loss:.4f}")
        
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}, best val loss: {best_val_loss:.4f}")
            break

    return best_val_loss

# Function to evaluate one hyperparameter combination over all CV folds
def evaluate_hyperparams(hparams, features, labels, batch_size, tscv, input_size, num_epochs, device, patience=5):
    lr = hparams['learning_rate']
    hidden = hparams['hidden_size']
    layers = hparams['num_layers']
    seq_len = hparams['seq_len']
    dropout = hparams['dropout']
    
    fold_losses = []
    for train_idx, val_idx in tscv.split(features):
        if len(train_idx) <= seq_len or len(val_idx) <= seq_len:
            continue
        train_loader, val_loader = create_dataloaders(features, labels, seq_len, batch_size, train_idx, val_idx)
        avg_loss = train_and_evaluate(
            train_loader, val_loader,
            input_size=input_size,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout,
            learning_rate=lr,
            num_epochs=num_epochs,
            device=device,
            patience=patience
        )
        fold_losses.append(avg_loss)
    avg_loss = np.mean(fold_losses) if fold_losses else None
    return {**hparams, 'avg_val_loss': avg_loss}

def main():
    # Force use GPU if available (you have one RTX 4080, so we'll use it)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    # Load configuration from config.yaml
    with open('src/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    data_path = config['data_path']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    
    # Load and preprocess data
    df = load_data(data_path)
    features, labels = select_features(df)
    features, scaler = normalize_features(features)
    features = np.array(features)
    labels = np.array(labels)
    input_size = features.shape[1]
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Define hyperparameter grids
    learning_rates = [0.001, 0.0005, 0.0001]
    hidden_sizes = [64, 128, 256]
    num_layers_options = [1, 2, 3]
    seq_lengths = [5, 10, 15]
    dropouts = [0.0, 0.2, 0.4]
    
    hyperparam_combinations = [
        {
            'learning_rate': lr,
            'hidden_size': hidden,
            'num_layers': layers,
            'seq_len': seq_len,
            'dropout': dropout
        }
        for lr in learning_rates
        for hidden in hidden_sizes
        for layers in num_layers_options
        for seq_len in seq_lengths
        for dropout in dropouts
    ]
    
    results = []
    # Sequential execution for GPU use
    for hparams in tqdm(hyperparam_combinations, desc="Hyperparameter Search"):
        result = evaluate_hyperparams(hparams, features, labels, batch_size, tscv, input_size, num_epochs, device, patience=5)
        if result['avg_val_loss'] is not None:
            results.append(result)
            print(f"lr: {result['learning_rate']}, hidden: {result['hidden_size']}, layers: {result['num_layers']}, seq_len: {result['seq_len']}, dropout: {result['dropout']}, avg_val_loss: {result['avg_val_loss']:.4f}")
        else:
            print("Skipped hyperparameter set due to insufficient data.")
    
    if results:
        best_result = min(results, key=lambda x: x['avg_val_loss'])
        print("\nBest Hyperparameters Found:")
        print(best_result)
    else:
        print("No valid hyperparameter combination was evaluated.")
    
    import pandas as pd
    results_df = pd.DataFrame(results)
    print("\nGrid Search Results:")
    print(results_df.sort_values(by='avg_val_loss'))
    
if __name__ == "__main__":
    main()
