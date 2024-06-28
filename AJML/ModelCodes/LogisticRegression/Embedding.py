import torch
import torch.nn as nn

class CategoricalEmbedding(nn.Module):
    def __init__(self, num_categories, embedding_dim):
        super(CategoricalEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim)

    def forward(self, x):
        return self.embedding(x)

def create_embedding_from_config(config):
    num_categories = config['num_categories']
    embedding_dim = config['embedding_dim']
    return CategoricalEmbedding(num_categories, embedding_dim)