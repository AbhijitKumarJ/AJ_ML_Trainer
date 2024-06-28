import torch
import torch.nn as nn

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_layers, num_classes):
        super(LogisticRegressionModel, self).__init__()
        
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
        
        layers.append(nn.Linear(prev_dim, num_classes))
        layers.append(nn.Softmax(dim=1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def create_model_from_config(config):
    input_dim = config['input_dim']
    hidden_layers = config['hidden_layers']
    num_classes = config['num_classes']
    return LogisticRegressionModel(input_dim, hidden_layers, num_classes)