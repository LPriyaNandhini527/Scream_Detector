import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Paths to dataset directories
SCREAM_DIR = r"C:\Users\Acer\Downloads\scream"
NON_SCREAM_DIR = r"C:\Users\Acer\Downloads\not scream"

# Function to extract MFCC features from audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# Custom Dataset class for scream detection
class ScreamDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Load dataset and extract features
def load_dataset():
    X, y = [], []
    for file in os.listdir(SCREAM_DIR):
        if file.endswith(".wav"):
            file_path = os.path.join(SCREAM_DIR, file)
            features = extract_features(file_path)
            X.append(features)
            y.append(1)  # Scream class

    for file in os.listdir(NON_SCREAM_DIR):
        if file.endswith(".wav"):
            file_path = os.path.join(NON_SCREAM_DIR, file)
            features = extract_features(file_path)
            X.append(features)
            y.append(0)  # Non-scream class

    X = np.array(X)
    y = np.array(y)
    return X, y

# Define CNN1D model
class CNN1DModel(nn.Module):
    def __init__(self):
        super(CNN1DModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 40, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define CNN2D model
class CNN2DModel(nn.Module):
    def __init__(self):
        super(CNN2DModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 10 * 4, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = x.view(-1, 1, 10, 4)  # Reshape for 2D CNN
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=40, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc1(x[:, -1, :])
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Training function
def train_model(model, optimizer, criterion, train_loader, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    accuracy = correct / total
    return accuracy

def main():
    # Load dataset
    X, y = load_dataset()

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = ScreamDataset(X_train, y_train)
    test_dataset = ScreamDataset(X_test, y_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize models
    cnn1d = CNN1DModel()
    cnn2d = CNN2DModel()
    lstm = LSTMModel()

    # Define loss and optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer_cnn1d = optim.Adam(cnn1d.parameters(), lr=0.001)
    optimizer_cnn2d = optim.Adam(cnn2d.parameters(), lr=0.001)
    optimizer_lstm = optim.Adam(lstm.parameters(), lr=0.001)

    # Train CNN1D
    print("Training CNN1D Model")
    train_model(cnn1d, optimizer_cnn1d, criterion, train_loader, epochs=10)
    acc_cnn1d = evaluate_model(cnn1d, test_loader)
    print(f"CNN1D Test Accuracy: {acc_cnn1d * 100:.2f}%")


    # Train CNN2D
    print("Training CNN2D Model")
    # For CNN2D, reshape input in the dataset loader or here
    def reshape_for_cnn2d(batch):
        X = torch.stack([item[0] for item in batch])
        y = torch.tensor([item[1] for item in batch])
        X = X.view(-1, 1, 10, 4)
        return X, y
    train_loader_cnn2d = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=reshape_for_cnn2d)
    test_loader_cnn2d = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=reshape_for_cnn2d)
    train_model(cnn2d, optimizer_cnn2d, criterion, train_loader_cnn2d, epochs=10)
    acc_cnn2d = evaluate_model(cnn2d, test_loader_cnn2d)
    print(f"CNN2D Test Accuracy: {acc_cnn2d * 100:.2f}%")

    # Train LSTM
    print("Training LSTM Model")
    def reshape_for_lstm(batch):
        X = torch.stack([item[0] for item in batch])
        y = torch.tensor([item[1] for item in batch])
        X = X.view(-1, 1, 40)
        return X, y
    train_loader_lstm = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=reshape_for_lstm)
    test_loader_lstm = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=reshape_for_lstm)
    train_model(lstm, optimizer_lstm, criterion, train_loader_lstm, epochs=10)
    acc_lstm = evaluate_model(lstm, test_loader_lstm)
    print(f"LSTM Test Accuracy: {acc_lstm * 100:.2f}%")

    # Save models
    torch.save(cnn1d.state_dict(), "scream_detector_cnn1d.pth")
    torch.save(cnn2d.state_dict(), "scream_detector_cnn2d.pth")
    torch.save(lstm.state_dict(), "scream_detector_lstm.pth")
    print("Models trained and saved successfully.")

if __name__ == "__main__":
    main()
