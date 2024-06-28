import torch
import torch.nn as nn
import torch.optim as optim
from Model import create_model_from_config, generate_square_subsequent_mask
from DataProcessor import load_and_preprocess_data

def train_model(config, data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, tokenizer = load_and_preprocess_data(data_path, config)
    model = create_model_from_config(config).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        for batch, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            src_mask = generate_square_subsequent_mask(src.size(0)).to(device)
            
            optimizer.zero_grad()
            output = model(src, src_mask)
            loss = criterion(output.view(-1, config['vocab_size']), tgt.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch % 200 == 0:
                print(f"Epoch {epoch}, Batch {batch}, Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch} completed. Average Loss: {total_loss / len(train_loader):.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                src_mask = generate_square_subsequent_mask(src.size(0)).to(device)
                output = model(src, src_mask)
                val_loss += criterion(output.view(-1, config['vocab_size']), tgt.view(-1)).item()
        
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}")
    
    return model, tokenizer

def save_model(model, tokenizer, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer
    }, path)

if __name__ == "__main__":
    import json
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    model, tokenizer = train_model(config, 'data.txt')
    save_model(model, tokenizer, 'gpt_basic_char_model.pth')