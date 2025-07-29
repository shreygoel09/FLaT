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

from utils.generate_utils import (
    calc_ppl, calc_entropy, calc_property_val,
    gen_random_tokens, get_logits, get_embeds, load_model_and_tokenizer, load_transport_model
)
from utils.model_utils import _print


os.chdir("/home/a03-sgoel/FLaT")
todays_date = date.today().strftime('%d-%m')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mode = "energy"  # energy / score
prop = "solubility"  # solubility / permeability / stability



class SVDD:
    """
    Soft Value-based Decoding in Diffusion models guidance method (no vmap version).
    """
    def __init__(self, alpha: float = 0.0, m: int = 10, maximize: bool = True):
        self.alpha = alpha
        self.m = m
        self.sign = 1.0 if maximize else -1.0

    def guide(self, x, diffusion, classifier, denoiser, dt, tokenizer):
        # Step 1: Repeat input x for m proto-samples
        all_gens = []
        all_rewards = []

        for i in range(self.m):
            # Copy inputs for current sample
            x_i = {}
            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    x_i[k] = v.clone()  # avoid reference sharing
                else:
                    x_i[k] = v

            # Step 2: Generate output
            gen_i = diffusion(...) #TODO: DPLM diffusion step
            all_gens.append(gen_i)

            # Step 3: Compute reward
            reward_i = classifier(clf_batch)
            all_rewards.append(self.sign * reward_i)

        # Collect all generated outputs and rewards
        gen = torch.stack(all_gens, dim=0)     # [m, B, ...]
        rewards = torch.stack(all_rewards, dim=0)  # [m, B]

        # Compute weights
        indices = rewards.argmax(dim=0)  # [B]

        # Step 6: Select final outputs
        gen = rearrange(gen, "m b ... -> b m ...")
        final_gen = torch.gather(
            gen,
            dim=1,
            index=indices.view(-1, 1, *([1] * (gen.ndim - 2))).expand(-1, 1, *gen.shape[2:])
        ).squeeze(1)

        return final_gen


# -------# Entry point for running #-------- #
def main():
    config = OmegaConf.load(f"./src/configs/{mode}/{prop}.yaml")

    tokenizer, encoder_lm_model, encoder_model = load_model_and_tokenizer(config, prop, device)
    transport_model = load_transport_model(mode, prop, config, device)

    seq_data = pd.read_csv(f"./data/{prop}/seqs_for_optim.csv")
    seqs = seq_data['Sequence'].tolist()
    
    perplexities = []
    og_vals = []
    optim_vals = []
    entropies = []

    for seq in tqdm(seqs, desc="Optimizing sequences"):

        masks = tokenizer.mask_token * len(seq)
        masked_seq = tokenizer(masks, return_tensors='pt')
        GuidedSampler = SVDD()

        attn_mask = torch.ones(1, len(seq)).to(device)
        
        #TODO: implement guidance here
        for t in range(500):
            logits = GuidedSampler.guide(...)

        # Sample tokens from categoricals
        optim_tokens = torch.distributions.Categorical(logits=logits).sample()
        optim_seq = tokenizer.decode(optim_tokens).replace(" ", "")[5:-5]
        _print(f'og seq:    {seq}')
        _print(f'optim seq: {optim_seq}')

        # Compute metrics
        try:
            perplexities.append(calc_ppl(encoder_lm_model, tokenizer, optim_seq))
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