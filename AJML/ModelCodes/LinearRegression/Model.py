import torch
import torch.nn as nn

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super(LinearRegressionModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for units, activation in hidden_layers:
            layers.append(nn.Linear(prev_dim, units))
            if activation.lower() == 'relu':
                layers.append(nn.ReLU())
            elif activation.lower() == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation.lower() == 'tanh':
                layers.append(nn.Tanh())
            prev_dim = units
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def create_model_from_config(config):
    input_dim = config['input_dim']
    hidden_layers = config['hidden_layers']
    output_dim = config['output_dim']
    return LinearRegressionModel(input_dim, hidden_layers, output_dim)
