#!/usr/bin/env python3

import os
import sys
import torch
import math
import importlib
import pandas as pd

from datetime import date
from tqdm import tqdm
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel

from utils.generate_utils import calc_ppl, calc_entropy, calc_hamming, calc_property_val
from utils.model_utils import _print


os.chdir("/home/a03-sgoel/FLaT")
todays_date = date.today().strftime('%d-%m')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -------# Pretrained model loading / usage #-------- #
def get_logits(tokens, model):
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        logits = model(**tokens).logits.squeeze(0)  # L, V
    return logits

def load_model_and_tokenizer(config):
    esm_path = config.lm.pretrained_esm
    tokenizer = AutoTokenizer.from_pretrained(esm_path)
    encoder_lm = AutoModelForMaskedLM.from_pretrained(esm_path).eval().to(device)
    
    encoder_model = AutoModel.from_pretrained(config.lm.pretrained_esm).eval().to(device)
    encoder_model.embeddings.token_dropout = False 

    return tokenizer, encoder_lm, encoder_model

def load_transport_model(mode: str, prop: str, config):
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



# -------# Langevin Transport #-------- #
def langevin_transport(z_t, transport_model, mode, eta, eps):
    """ Function to perform 1 step of latent Langevin transport """
    noise = math.sqrt(2 * eps) * torch.randn_like(z_t, device=z_t.device)
    z_t = z_t.clone().detach().requires_grad_(True)
    
    if mode == "energy":
        energy_vector = transport_model({"embeds": z_t})
        energy_vector.backward()  # energy is already scalar
        z_t = z_t - (eta * z_t.grad) + noise
    
    elif mode == "score":
        with torch.no_grad():
            s_theta, _, _ = transport_model({"embeds": z_t})
        _print(f'score: {s_theta}')
        _print(f'zt: {z_t}')
        z_t = z_t + (eta * s_theta) + noise

    return z_t.detach()

# -------# Jacobian Decoding #-------- #
def jacobian_decoding(x_k, optim_latent, encoder, eta):
    """
    Function to perform 1 step of vector-jacobian product decoding.
    
    Args:
        - x_k (torch.Tensor): sequence logits [L, V]
        - z_target (torch.Tensor): langevin-transported latent (D)
        - encoder (PreTrainedModel): pre-trained encoder model. Must have get_input_embeddings() fn
        - eta: learning rate / update step size
    Return:
        - x_prime (torch.Tensor): optimzed  
    """
    x_k = x_k.detach().clone().requires_grad_(True)
    x_flat = x_k.view(-1)

    def relax_fn(x_flat):
        # Create soft (differentiable) embeddings
        x = x_flat.view(x_k.shape) # L, V
        probs = x.softmax(dim=-1).unsqueeze(0) # 1, L, V
        
        # Convert from token to embedding space
        word_embeds = torch.matmul(probs, encoder.get_input_embeddings().weight) # 1, L, D
        
        # Compute hidden states of relaxed sequence
        word_embeds *= 1 - (0.15 * 0.8) # bypass EsmEmbedding token dropout logic
        outs = encoder(inputs_embeds=word_embeds, output_hidden_states=True)
        embeds = outs.hidden_states[-1].mean(dim=1) # 1, D
        return embeds

    # Residual vector
    pi_x = relax_fn(x_flat)
    residual = pi_x - optim_latent # D
    
    # Vector-Jacobian product of encoder model wrt hidden states
    # d_pi(x) / dx * (pi(x)) - z')
    grad = torch.autograd.grad(
        outputs=pi_x,
        inputs=x_flat,
        grad_outputs=residual, # Vector is the residuals
        retain_graph=False
    )[0]

    x_k_prime = x_flat - (eta * grad)
    logits = x_k_prime.view(x_k.shape).detach() # L, V
    return logits



# -------# Entry point for running #-------- #
def main():
    mode = "score"  # energy / score
    prop = "solubility"  # solubility / permeability / stability

    config = OmegaConf.load(f"./src/configs/{mode}/{prop}.yaml")

    tokenizer, encoder_lm_model, encoder_model = load_model_and_tokenizer(config)
    transport_model = load_transport_model(mode, prop, config)

    seq_data = pd.read_csv(f"./data/{prop}/seqs_for_optim.csv")
    seqs = seq_data['Sequence'].tolist()
    
    perplexities = []
    entropies = []
    optim_vals = []
    hammings = []

    for seq in tqdm(seqs, desc="Optimizing sequences"):
        
        # Perform langevin transport on the initial vector of gaussian noise
        latent = torch.randn(1, config.model.d_model).to(device)
        langevin_lr = torch.tensor(config.sampling.langevin.lr, device=device, dtype=torch.float32)
        langevin_eps = torch.tensor(config.sampling.langevin.noise_eps, device=device, dtype=torch.float32)
        for _ in range(config.sampling.langevin.steps):
            latent = langevin_transport(
                z_t=latent,
                transport_model=transport_model,
                mode=mode,
                eta=langevin_lr,
                eps=langevin_eps
            )
        
        # Compute and optimize sequence logits
        tokens = tokenizer(seq, return_tensors='pt')
        logits = get_logits(tokens, encoder_lm_model)
        decoder_lr = torch.tensor(config.sampling.decoding.lr, device=device, dtype=torch.float32)
        for _ in range(config.sampling.decoding.steps):
            logits = jacobian_decoding(
                x_k=logits,
                optim_latent=latent,
                encoder=encoder_model,
                eta=decoder_lr
            )

        # Sample tokens from categoricals
        optim_tokens = torch.distributions.Categorical(logits=logits).sample()
        optim_seq = tokenizer.decode(optim_tokens).replace(" ", "")[5:-5]
        _print(f'og seq: {seq}')
        _print(f'optim seq: {optim_seq}')

        # Compute metrics
        perplexities.append(calc_ppl(encoder_lm_model, tokenizer, optim_seq))
        entropies.append(calc_entropy(optim_seq))
        optim_vals.append(calc_property_val(optim_seq, tokenizer, transport_model, mode))
        hammings.append(calc_hamming(seq, optim_seq))

    # Save results
    seq_data['PPL'] = perplexities
    seq_data['entropy'] = entropies
    
    save_path = f'./results/optim/{mode}/{prop}/{todays_date}_optim_seqs.csv'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    seq_data.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()






# # -------# Jacobian Decoding #-------- #
# def jacobian_decoding(x_k, optim_latent, encoder, eta):
#     """
#     Function to perform 1 step of Jacobian-informed pseudo-inverse decoding.
    
#     Args:
#         - x_k (torch.Tensor): sequence logits [L, V]
#         - z_target (torch.Tensor): langevin-transported latent (D)
#         - encoder (PreTrainedModel): pre-trained encoder model. Must have get_input_embeddings() fn
#         - eta: learning rate / update step size
#     Return:
#         - x_prime (torch.Tensor): optimzed  
#     """
#     x_k = x_k.detach().clone().requires_grad_(True)

#     def relax_fn(x_flat):
#         # Create soft (differentiable) embeddings
#         x = x_flat.view(x_k.shape) # L, V
#         probs = x.softmax(dim=-1).unsqueeze(0) # 1, L, V
        
#         # Convert from token to embedding space
#         word_embeds = torch.matmul(probs, encoder.get_input_embeddings().weight) # 1, L, D
#         word_embeds *= 0.15 * 0.8
        
#         # Compute hidden states of relaxed sequence
#         outs = encoder(inputs_embeds=word_embeds, output_hidden_states=True)
#         embeds = outs.hidden_states[-1].mean(dim=1).squeeze(0) # D
#         return embeds

#     x_flat = x_k.flatten()
#     jacobian = torch.autograd.functional.jacobian(
#         func=relax_fn,
#         inputs=x_flat,
#         create_graph=False
#     )
#     j_pinv = torch.linalg.pinv(jacobian) # L x V, D

#     residual = relax_fn(x_flat) - optim_latent # D
#     update = torch.matmul(j_pinv, residual) # L x V
#     x_prime = x_flat - (eta.squeeze(0) * update)

#     return x_prime.view(x_k.shape).detach() # L, V


