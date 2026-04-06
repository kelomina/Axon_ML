import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
from settings import GATING_HIDDEN_DIM, GATING_EPOCHS, GATING_BATCH_SIZE, GATING_LEARNING_RATE, MODEL_DIR

class GatingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_gating_model(X_train, y_train, X_val, y_val, model_path, input_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GatingMLP(input_dim, GATING_HIDDEN_DIM, 2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=GATING_LEARNING_RATE)
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=GATING_BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=GATING_BATCH_SIZE)
    best_acc = 0.0
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    for _ in range(GATING_EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_path)
    return best_acc

def load_gating_model(model_path, input_dim):
    model = GatingMLP(input_dim, GATING_HIDDEN_DIM, 2)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        if not _validate_model_state_dict(state_dict, model):
            print(f"[Warning] Model state_dict structure mismatch, using untrained model")
        else:
            model.load_state_dict(state_dict)
    else:
        print(f"[Warning] Gating model not found at {model_path}, using untrained model")
    model.eval()
    return model

def _validate_model_state_dict(state_dict, model):
    if not isinstance(state_dict, dict):
        return False
    model_state_keys = set(model.state_dict().keys())
    loaded_keys = set(state_dict.keys())
    if model_state_keys != loaded_keys:
        missing = model_state_keys - loaded_keys
        extra = loaded_keys - model_state_keys
        if missing:
            print(f"[Warning] Missing keys in state_dict: {missing}")
        if extra:
            print(f"[Warning] Extra keys in state_dict: {extra}")
        return False
    return True

def predict_gating(model, X):
    with torch.no_grad():
        logits = model(torch.FloatTensor(X))
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return probs
