# Note: For linear regression, we typically don't need a tokenizer.
# This file is included for consistency with more complex models.

def tokenize(text):
    # For linear regression, we might just return the text as-is
    return text

def detokenize(tokens):
    # For linear regression, we might just join the tokens
    return ' '.join(tokens) if isinstance(tokens, list) else tokens
