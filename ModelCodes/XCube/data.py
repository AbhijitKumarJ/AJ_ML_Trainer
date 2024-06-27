import torch

def generate_data():
    X = torch.linspace(-5, 5, 200).reshape(-1, 1)
    y = X**3 # + torch.randn(200, 1) * 2

    # Normalize the data
    X = (X - X.mean()) / X.std()
    y = (y - y.mean()) / y.std()
        
    return X, y