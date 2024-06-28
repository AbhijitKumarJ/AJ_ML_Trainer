import json

class CharTokenizer:
    def __init__(self, config):
        self.char_to_idx = {char: idx for idx, char in enumerate(config['vocab'])}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

    def encode(self, text):
        return [self.char_to_idx.get(char, self.char_to_idx['<unk>']) for char in text]

    def decode(self, tokens):
        return ''.join([self.idx_to_char.get(idx, '<unk>') for idx in tokens])

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'char_to_idx': self.char_to_idx}, f)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        tokenizer = cls({'vocab': list(data['char_to_idx'].keys())})
        tokenizer.char_to_idx = data['char_to_idx']
        tokenizer.idx_to_char = {idx: char for char, idx in tokenizer.char_to_idx.items()}
        return tokenizer

def create_tokenizer_from_config(config):
    return CharTokenizer(config)
