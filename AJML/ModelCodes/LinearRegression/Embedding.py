import torch
import torch.nn as nn

class LinearEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(LinearEmbedding, self).__init__()
        self.embedding = nn.Linear(input_dim, embedding_dim)

    def forward(self, x):
        return self.embedding(x)

def create_embedding_from_config(config):
    input_dim = config['input_dim']
    embedding_dim = config['embedding_dim']
    return LinearEmbedding(input_dim, embedding_dim)