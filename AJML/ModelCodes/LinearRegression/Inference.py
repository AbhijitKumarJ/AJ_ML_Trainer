import torch
from Model import create_model_from_config
from DataProcessor import load_and_preprocess_data

def load_model(model_path, config):
    model = create_model_from_config(config)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, X):
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        predictions = model(X_tensor)
    return predictions.numpy()

def run_inference(model_path, config_path, data_path, target_column):
    config = load_config(config_path)
    model = load_model(model_path, config)
    X, _ = load_and_preprocess_data(data_path, target_column)
    predictions = predict(model, X)
    return predictions

def load_config(config_path):
    import json
    with open(config_path, 'r') as f:
        return json.load(f)
