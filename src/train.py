import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model import FinReportModel
from data_loader import load_data, split_data
from preprocessing import select_features, normalize_features

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
dropout = model_config.get('dropout', 0.0)        # default to 0.0 if not provided

print("Loaded configuration:")
print(config)

# Load and preprocess data
df = load_data(data_path)
train_df, test_df = split_data(df)
train_features, train_labels = select_features(train_df)
test_features, test_labels = select_features(test_df)
train_features, scaler = normalize_features(train_features)
test_features = scaler.transform(test_features)

# Define the dataset class (for sequence data)
class FinDataset(Dataset):
    def __init__(self, features, labels, seq_len):
        self.features = features
        self.labels = labels
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.seq_len]
        y = self.labels[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)

train_dataset = FinDataset(train_features, train_labels, seq_len)
test_dataset = FinDataset(test_features, test_labels, seq_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model with parameters from the config
model = FinReportModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Training loop
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
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), 'finreport_model.pth')
print("Model saved to finreport_model.pth")
