import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

X = pd.read_csv('features.csv')
y = pd.read_csv('outputs.csv')

X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

X_tensor = torch.FloatTensor(X_scaled)
y_tensor = torch.FloatTensor(y_scaled)

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
        super(FeedForwardNN, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.Dropout(dropout_rate))
        
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(nn.Dropout(dropout_rate))
        
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.3, random_state=42)

train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

input_size = X.shape[1]
hidden_sizes = [256, 128, 64]
output_size = y.shape[1]

model = FeedForwardNN(input_size, hidden_sizes, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6, eps=1e-8)

# Cosine annealing scheduler w/ warm restarts
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-6
)

num_epochs = 500
train_losses = []
val_losses = []
best_loss = float('inf')
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    epoch_losses = []
    
    for _ in range(2):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            epoch_losses.append(loss.item())
    
    avg_train_loss = np.mean(epoch_losses)
    train_losses.append(avg_train_loss)
    
    model.eval()
    val_epoch_losses = []
    all_val_preds = []
    all_val_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            val_loss = criterion(outputs, batch_y)
            val_epoch_losses.append(val_loss.item())
            all_val_preds.append(outputs)
            all_val_targets.append(batch_y)
    
    avg_val_loss = np.mean(val_epoch_losses)
    val_losses.append(avg_val_loss)
    
    # Save best model
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        best_model_state = model.state_dict().copy()
    
    scheduler.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')

# Load
model.load_state_dict(best_model_state)

# Save
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'X_scaler': X_scaler,
    'y_scaler': y_scaler,
    'train_losses': train_losses,
    'val_losses': val_losses,
    'best_loss': best_loss,
    'hidden_sizes': hidden_sizes
}, 'neural_network_model.pth')

model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_orig = y_scaler.inverse_transform(y_pred.numpy())
    y_test_orig = y_scaler.inverse_transform(y_test.numpy())
    
for i in range(output_size):
    r2 = r2_score(y_test_orig[:, i], y_pred_orig[:, i])
    mse = np.mean((y_test_orig[:, i] - y_pred_orig[:, i])**2)
    mae = np.mean(np.abs(y_test_orig[:, i] - y_pred_orig[:, i]))
    print(f'\nMetrics for output {i+1} ({y.columns[i]}):')
    print(f'R² score: {r2:.3f}')
    print(f'MSE: {mse:.3f}')
    print(f'MAE: {mae:.3f}')