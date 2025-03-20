import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from model import FinReportModel
from data_loader import load_data, split_data
from preprocessing import select_features, normalize_features

# Create only necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('img', exist_ok=True)

# Early Stopping implementation
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            trace_func (function): trace print function.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        
    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def plot_learning_curves(train_losses, val_losses):
    """Plot the training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig('img/learning_curves.png')
    plt.close()
    print("Learning curves saved to 'img/learning_curves.png'")

# Load configuration from config.yaml
with open('src/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract hyperparameters from the config
data_path = config['data_path']
batch_size = config['batch_size']
seq_len = config['seq_len']
learning_rate = config['learning_rate']
num_epochs = config['num_epochs']
model_config = config['model']
input_size = model_config['input_size']       # Make sure this matches your feature count
hidden_size = model_config['hidden_size']
num_layers = model_config.get('num_layers', 1)  # default to 1 if not provided
dropout = model_config.get('dropout', 0.0)      # default to 0.0 if not provided

# Add validation parameters
val_ratio = 0.2  # 20% of data for validation
patience = 7     # Early stopping patience

print("Loaded configuration:")
print(config)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load and preprocess data
print("Loading data...")
df = load_data(data_path)
train_df, test_df = split_data(df)
train_features, train_labels = select_features(train_df)
test_features, test_labels = select_features(test_df)

# Normalize features
train_features, scaler = normalize_features(train_features)
test_features = scaler.transform(test_features)

# Define the dataset class (for sequence data)
class FinDataset(Dataset):
    def __init__(self, features, labels, seq_len):
        # Ensure features and labels are of the same length
        min_len = min(len(features), len(labels))
        self.features = features[:min_len]
        self.labels = labels[:min_len]
        self.seq_len = seq_len

    def __len__(self):
        # Only return valid indices where we can construct a full sequence
        return max(0, len(self.features) - self.seq_len)

    def __getitem__(self, idx):
        # Get a sequence of length seq_len starting at idx
        x = self.features[idx:idx + self.seq_len]
        
        # Get the label that follows the sequence
        y = self.labels[idx + self.seq_len - 1]  # Use the last element of the sequence window
        
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)

# Create full dataset
full_dataset = FinDataset(train_features, train_labels, seq_len)

# Split into train and validation
dataset_size = len(full_dataset)
val_size = int(val_ratio * dataset_size)
train_size = dataset_size - val_size

# Create train and validation datasets
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = FinDataset(test_features, test_labels, seq_len)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model with parameters from the config
model = FinReportModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
model = model.to(device)

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model has {trainable_params:,} trainable parameters")

# Initialize optimizer (without weight decay since dropout=0.0 was optimal)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize loss function
criterion = nn.MSELoss()

# Initialize learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, min_lr=0.00001, verbose=True
)

# Initialize early stopping
early_stopping = EarlyStopping(
    patience=patience,
    verbose=True,
    delta=0.0001,
    path='models/best_model.pt'
)

# Training loop with validation
train_losses = []
val_losses = []
learning_rates = []

start_time = time.time()
print("Starting training...")

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    
    # Training phase
    model.train()
    train_loss = 0.0
    
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        # Apply gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_loss += loss.item() * x_batch.size(0)
    
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            
            outputs = model(x_val)
            loss = criterion(outputs, y_val)
            
            val_loss += loss.item() * x_val.size(0)
    
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    
    # Update learning rate
    scheduler.step(val_loss)
    
    # Store current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)
    
    # Early stopping check
    early_stopping(val_loss, model)
    
    # Print epoch statistics
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Time: {epoch_time:.2f}s | "
          f"Train Loss: {train_loss:.6f} | "
          f"Val Loss: {val_loss:.6f} | "
          f"LR: {current_lr:.6f}")
    
    if early_stopping.early_stop:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

# Calculate total training time
total_time = time.time() - start_time
print(f"Training completed in {total_time:.2f} seconds")

# Load best model
model.load_state_dict(torch.load('models/best_model.pt'))

# Evaluate on test set
model.eval()
test_loss = 0.0
all_test_preds = []
all_test_targets = []

with torch.no_grad():
    for x_test, y_test in test_loader:
        x_test, y_test = x_test.to(device), y_test.to(device)
        
        outputs = model(x_test)
        loss = criterion(outputs, y_test)
        
        test_loss += loss.item() * x_test.size(0)
        all_test_preds.extend(outputs.cpu().numpy())
        all_test_targets.extend(y_test.cpu().numpy())

test_loss /= len(test_loader.dataset)
print(f"Test Loss: {test_loss:.6f}")

# Calculate regression metrics
all_test_preds = np.array(all_test_preds)
all_test_targets = np.array(all_test_targets)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
test_mse = mean_squared_error(all_test_targets, all_test_preds)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(all_test_targets, all_test_preds)
test_r2 = r2_score(all_test_targets, all_test_preds)

print(f"Test MSE: {test_mse:.6f}")
print(f"Test RMSE: {test_rmse:.6f}")
print(f"Test MAE: {test_mae:.6f}")
print(f"Test R²: {test_r2:.6f}")

# Save model
torch.save(model.state_dict(), 'models/finreport_model.pth')
print("Model saved to models/finreport_model.pth")

# Plot learning curves
plt.figure(figsize=(12, 8))

# Loss plot
plt.subplot(2, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Learning rate plot
plt.subplot(2, 2, 2)
plt.plot(learning_rates)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.grid(True)

# Predictions vs Actual plot
plt.subplot(2, 2, 3)
plt.scatter(all_test_targets, all_test_preds, alpha=0.3)
min_val = min(np.min(all_test_targets), np.min(all_test_preds))
max_val = max(np.max(all_test_targets), np.max(all_test_preds))
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Predictions vs Actual (R² = {test_r2:.4f})')
plt.grid(True)

# Error distribution plot
plt.subplot(2, 2, 4)
errors = all_test_preds - all_test_targets
plt.hist(errors, bins=30)
plt.axvline(x=0, color='r', linestyle='--')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title(f'Error Distribution (RMSE = {test_rmse:.4f})')
plt.grid(True)

plt.tight_layout()
plt.savefig('img/training_results.png')
plt.close()

print("Training results plots saved to img/training_results.png")
print("Training script completed successfully")

def perform_cross_validation(df, model_config, k_folds=5):
    from sklearn.model_selection import TimeSeriesSplit
    # Extract parameters
    input_size = model_config['input_size']
    hidden_size = model_config['hidden_size']
    num_layers = model_config.get('num_layers', 1)
    dropout = model_config.get('dropout', 0.0)
    seq_len = config['seq_len']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    
    # Extract features and labels
    features, labels = select_features(df)
    features, scaler = normalize_features(features)
    
    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=k_folds)
    
    # Store metrics for each fold
    fold_metrics = []
    
    # Create figure for plots
    plt.figure(figsize=(15, 10))
    
    # Perform k-fold cross-validation
    for fold, (train_idx, test_idx) in enumerate(tscv.split(features)):
        print(f"Training fold {fold+1}/{k_folds}")
        
        # Split data
        train_features, test_features = features[train_idx], features[test_idx]
        train_labels, test_labels = labels[train_idx], labels[test_idx]
        
        # Create datasets
        train_dataset = FinDataset(train_features, train_labels, seq_len)
        test_dataset = FinDataset(test_features, test_labels, seq_len)
        
        if len(train_dataset) == 0 or len(test_dataset) == 0:
            print(f"Skipping fold {fold+1} due to insufficient data")
            continue
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model = FinReportModel(input_size=input_size, hidden_size=hidden_size, 
                               num_layers=num_layers, dropout=dropout)
        model = model.to(device)
        
        # Initialize optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Train model (using fewer epochs for CV)
        for epoch in range(10):
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
        
        # Evaluate model
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test, y_test = x_test.to(device), y_test.to(device)
                outputs = model(x_test)
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(y_test.cpu().numpy())
        
        # Calculate regression metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        mse = mean_squared_error(all_labels, all_preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds)
        
        # Plot predictions vs actual for this fold
        plt.subplot(k_folds, 2, fold*2+1)
        plt.scatter(all_labels, all_preds, alpha=0.5)
        plt.plot([min(all_labels), max(all_labels)], [min(all_labels), max(all_labels)], 'r--')
        plt.title(f'Fold {fold+1}: Predictions vs Actual (R² = {r2:.4f})')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        
        # Plot distribution of predictions and labels
        plt.subplot(k_folds, 2, fold*2+2)
        plt.hist(all_labels, bins=20, alpha=0.5, label='Actual')
        plt.hist(all_preds, bins=20, alpha=0.5, label='Predicted')
        plt.title(f'Fold {fold+1}: Distribution (RMSE = {rmse:.4f})')
        plt.legend()
        
        # Store metrics
        fold_metrics.append({
            'fold': fold+1,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        })
    
    plt.tight_layout()
    plt.savefig('img/cross_validation_results.png')
    plt.close()
    
    # Print average metrics
    if fold_metrics:
        avg_mse = np.mean([m['mse'] for m in fold_metrics])
        avg_rmse = np.mean([m['rmse'] for m in fold_metrics])
        avg_mae = np.mean([m['mae'] for m in fold_metrics])
        avg_r2 = np.mean([m['r2'] for m in fold_metrics])
        
        print("\nCross-Validation Results:")
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"Average RMSE: {avg_rmse:.6f}")
        print(f"Average MAE: {avg_mae:.6f}")
        print(f"Average R²: {avg_r2:.6f}")
    
    return fold_metrics