import torch
from Model import create_model_from_config, generate_square_subsequent_mask
from Tokenizer import CharTokenizer

def load_model(model_path, config):
    checkpoint = torch.load(model_path)
    model = create_model_from_config(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    tokenizer = checkpoint['tokenizer']
    return model, tokenizer

def generate_text(model, tokenizer, start_text, max_length, temperature=1.0):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    input_ids = torch.tensor(tokenizer.encode(start_text)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            src_mask = generate_square_subsequent_mask(input_ids.size(1)).to(device)
            output = model(input_ids, src_mask)
            
            # Apply temperature
            logits = output[:, -1, :] / temperature
            next_token_probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if next_token.item() == tokenizer.char_to_idx.get('<eos>', 0):
                break
    
    return tokenizer.decode(input_ids.squeeze().tolist())

def run_inference(model_path, config_path, start_text, max_length, temperature=1.0):
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model, tokenizer = load_model(model_path, config)
    generated_text = generate_text(model, tokenizer, start_text, max_length, temperature)
    return generated_text

if __name__ == "__main__":
    generated_text = run_inference('gpt_basic_char_model.pth', 'config.json', "Once upon a time", 100)
    print(generated_text)