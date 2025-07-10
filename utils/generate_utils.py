import math
import torch

def calc_ppl(model, tokenizer, generated_sequence):
    total_loss = 0.0
    tensor_input = tokenizer.encode(generated_sequence, return_tensors='pt').to(model.device)

    for i in range(len(generated_sequence)):
        masked_input = tensor_input.clone()
        masked_input[0, i] = tokenizer.mask_token_id
    
        labels = torch.full(tensor_input.shape, -100).to(model.device)
        labels[0, i] = tensor_input[0, i]

        with torch.no_grad():
            loss = model(masked_input, labels=labels).loss.item()
        
        total_loss += loss
    
    avg_loss = total_loss / len(generated_sequence)
    perplexity = math.exp(avg_loss)

    return perplexity