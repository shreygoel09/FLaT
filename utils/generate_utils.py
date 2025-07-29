import math
import torch
import importlib
from collections import Counter
from transformers import AutoModelForMaskedLM, AutoModel, AutoTokenizer
from utils.model_utils import _print


# -------# Sampling pretrained model loading / usage #-------- #
def gen_random_tokens(seq_len, prop, device):
    if prop in {"solubility", "stability"}:
        ids = torch.randint(4, 23, (1, seq_len + 2), device=device)
        masks = torch.ones_like(ids)
        tokens = {'input_ids': ids, 'attention_mask': masks}
        return tokens

def get_logits(tokens, model):
    tokens = {k: v.to(model.device) for k, v in tokens.items()}
    with torch.no_grad():
        logits = model(**tokens).logits.squeeze(0)  # L, V
    return logits

def get_embeds(tokens, model):
    tokens = {k: v.to(model.device) for k, v in tokens.items()}
    with torch.no_grad():
        embeds = model(**tokens, output_hidden_states=True).hidden_states[-1]
    return embeds

def load_evoflow(config, device):
    path = "fredzzp/EvoFlow-650M-context-3070"
    tokenizer = AutoTokenizer.from_pretrained(path)
    encoder_lm = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D").eval().to(device)
    encoder_model = AutoModel.from_pretrained(path).eval().to(device)
    return tokenizer, encoder_lm, encoder_model


def load_model_and_tokenizer(config, prop, device):
    if prop in {"solubility", "stability"}:
        esm_path = config.lm.pretrained_esm
        tokenizer = AutoTokenizer.from_pretrained(esm_path)
        encoder_lm = AutoModelForMaskedLM.from_pretrained(esm_path).eval().to(device)
        
        encoder_model = AutoModel.from_pretrained(config.lm.pretrained_esm).eval().to(device)
        encoder_model.embeddings.token_dropout = False 

        return tokenizer, encoder_lm, encoder_model

def load_transport_model(mode: str, prop: str, config, device):
    guidance_models = importlib.import_module(f"src.latent_transport.{mode}.models")
    if prop == "solubility":
        guidance_model = guidance_models.SolubilityClassifier(config)
    elif prop == "stability":
        guidance_model = guidance_models.StabilityRegressor(config)
    elif prop == "permeability":
        guidance_model = guidance_models.PermeabiltyRegressor(config)

    pl_module = importlib.import_module(f"src.latent_transport.{mode}.{prop}.pl_module")
    transport_model = pl_module.TransportModule(config, guidance_model)
    
    state_dict = torch.load(config.checkpointing.best_ckpt_path)['state_dict']
    transport_model.load_state_dict(state_dict)

    return transport_model.to(device).eval()


# -------# Metrics calculation #-------- #
def calc_property_val(seq, tokenizer, transport_model, mode, device):
    tokens = tokenizer(seq, return_tensors='pt').to(device)
    if mode == "score":
        score, z, z_prime = transport_model(tokens)
        metric = score.norm(p=2, dim=1) / score.numel()
    elif mode == "energy":
        logits = transport_model(tokens)
        _print(f'logits: {logits}')
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