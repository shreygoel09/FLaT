import math
import torch
from collections import Counter
from utils.model_utils import _print

def calc_property_val(seq, tokenizer, transport_model, mode):
    tokens = tokenizer(seq, return_tensors='pt')
    if mode == "score":
        score, z, z_prime = transport_model(**tokens)
        metric = score.norm(p=2, dim=1) / score.numel()
    elif mode == "energy":
        logits = transport_model(**tokens)
        metric = torch.sigmoid(logits)
    return metric


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

def calc_hamming(seq1, seq2):
    assert len(seq1) == len(seq2), f'Seq lens are different ({len(seq1)}, {len(seq2)})'
    diffs = sum(char1 != char2 for char1, char2 in zip(seq1, seq2))
    return diffs / len(seq1) # Normalize for interpretability


def calc_entropy(seq):
    counts = Counter(seq)
    entropy = 0.0
    for residue, count in counts.items():
        prob = count / len(seq)
        entropy -= prob * math.log2(prob)
    return entropy