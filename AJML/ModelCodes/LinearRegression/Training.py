import torch
import torch.nn as nn
import torch.optim as optim
from Model import create_model_from_config
from DataProcessor import load_and_preprocess_data, create_data_loaders

def train_model(config, data_path, target_column, epochs=100, learning_rate=0.01):
    model = create_model_from_config(config)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    X, y = load_and_preprocess_data(data_path, target_column)
    train_loader, test_loader = create_data_loaders(X, y, batch_size=32)

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = sum(criterion(model(batch_X), batch_y.unsqueeze(1)) for batch_X, batch_y in test_loader)
            val_loss /= len(test_loader)

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

    return model

def save_model(model, path):
    torch.save(model.state_dict(), path)

if __name__ == "__main__":
    import json
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    model = train_model(config, 'data.csv', 'target')
    save_model(model, 'linear_regression_model.pth')