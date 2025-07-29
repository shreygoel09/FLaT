#!/usr/bin/env python3

import os
import sys
import torch
import math
import pandas as pd
import torch.nn.functional as F

from einops import rearrange
from reinmax import reinmax
from datetime import date
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.optim import Adagrad

from p2_sampler import stochastic_sample_from_categorical
from utils.generate_utils import (
    calc_ppl, calc_entropy, calc_property_val,
    gen_random_tokens, get_logits, get_embeds, load_evoflow, load_transport_model
)
from utils.model_utils import _print


os.chdir("/home/a03-sgoel/FLaT")
todays_date = date.today().strftime('%d-%m')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mode = "energy"  # energy / score
prop = "solubility"  # solubility / permeability / stability



class NOS:
    """
    Implementation of: https://arxiv.org/pdf/2305.20009
    Adapted from: https://github.com/ngruver/NOS/blob/main/seq_models/model/mlm_diffusion.py
    """
    def __init__(self, step_size=1.0, stability_coeff=1e-2, k=10, maximize=True, reinmax_temperature=1.0):
        self.k = k
        self.step_size = step_size
        self.stability_coeff = stability_coeff
        self.sign = -1.0 if maximize else 1.0
        self.kl = torch.nn.KLDivLoss(log_target=True)
        self.reinmax_temperature = reinmax_temperature

    def requires_grad(self):
        return True

    def sample(self, x, diffusion, encoder, classifier, attn_mask):

        # Initial embeddings and logits
        logits = diffusion(x).double()
        
        # Initialize optimizer to store gradients
        h = F.softmax(logits / self.reinmax_temperature, dim=-1)
        h.requires_grad_(True)
        
        delta = torch.zeros_like(h, requires_grad=True)
        optimizer = Adagrad([delta], lr=self.step_size)

        with torch.enable_grad():
            for _ in range(self.k):
                # Update logits and sample the corresponding new tokens
                h_current = h + delta
                tokens, _ = stochastic_sample_from_categorical(h_current)
                
                # Forward pass
                new_logits = diffusion(tokens)
                new_embeds = encoder({'input_ids': tokens, "attention_mask": attn_mask})
                
                # Langevin update loss
                kl_loss = self.kl(new_logits.log_softmax(dim=-1), logits.log_softmax(dim=-1)) # KL divergence
                reward_loss = classifier(new_embeds) # reward loss
                loss = self.stability_coeff * kl_loss + self.sign * reward_loss
                
                # Gradient step
                optimizer.zero_grad()
                loss.sum().backward()
                optimizer.step()
        
        final_logits = diffusion(h + delta).double
        return final_logits




# -------# Entry point for running #-------- #
def main():
    config = OmegaConf.load(f"./src/configs/{mode}/{prop}.yaml")

    tokenizer, diffusion, diffusion_encoder = load_evoflow(config, device)
    transport_model = load_transport_model(mode, prop, config, device)

    seq_data = pd.read_csv(f"./data/{prop}/seqs_for_optim.csv")
    seqs = seq_data['Sequence'].tolist()
    
    perplexities = []
    og_vals = []
    optim_vals = []
    entropies = []

    for seq in tqdm(seqs, desc="Optimizing sequences"):

        attn_mask = torch.ones((1, len(seq)), device=device)
        
        # Sample initial sequence
        xt = tokenizer(tokenizer.mask_token * len(seq), return_tensors='pt')
        logits = diffusion({'input_ids': xt, 'attention_mask': attn_mask})
        
        GuidedSampler = NOS()

        for _ in range(config.guidance.decoding.steps):
            logits = GuidedSampler.sample(
                x=logits,
                diffusion=diffusion,
                encoder=diffusion_encoder,
                classifier=transport_model,
                attn_mask=attn_mask
            )

        # Sample tokens from categoricals
        optim_tokens, _ = stochastic_sample_from_categorical(logits)
        optim_seq = tokenizer.decode(optim_tokens).replace(" ", "")[5:-5]
        _print(f'og seq:    {seq}')
        _print(f'optim seq: {optim_seq}')

        # Compute metrics
        try:
            perplexities.append(calc_ppl(diffusion, tokenizer, optim_seq))
            og_vals.append(calc_property_val(seq, tokenizer, transport_model, mode, device).item())
            optim_vals.append(calc_property_val(optim_seq, tokenizer, transport_model, mode, device).item())
            entropies.append(calc_entropy(optim_seq))
        except:
            perplexities.append(float('inf'))
            og_vals.append(float('inf'))
            optim_vals.append(float('inf'))
            entropies.append(float('inf'))

        _print(f'ppls:       {perplexities}')
        _print(f'og vals:    {og_vals}')
        _print(f'optim vals: {optim_vals}')
        _print(f'entropy:    {entropies}')
        _print('\n')

    # Save results
    seq_data['PPL'] = perplexities
    seq_data['OG Vals'] = og_vals
    seq_data['Optim Vals'] = optim_vals
    seq_data['entropy'] = entropies
    
    save_path = f'./results/svdd/{mode}/{prop}/{todays_date}/transformer_optim_seqs.csv'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    seq_data.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()