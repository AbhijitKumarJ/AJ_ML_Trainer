import torch
import torch.nn as nn

class CharEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CharEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)

def create_embedding_from_config(config):
    return CharEmbedding(config['vocab_size'], config['d_model'])