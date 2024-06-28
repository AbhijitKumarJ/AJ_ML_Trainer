import torch
from torch.utils.data import Dataset, DataLoader
from Tokenizer import create_tokenizer_from_config

class CharDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len):
        self.text = text
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data = self.tokenizer.encode(self.text)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.data[idx:idx+self.seq_len+1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def load_and_preprocess_data(file_path, config):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    tokenizer = create_tokenizer_from_config(config)
    dataset = CharDataset(text, tokenizer, config['seq_len'])
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)

    return train_loader, val_loader, tokenizer

def process_file_for_tokenization(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text
