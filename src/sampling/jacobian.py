import os
import sys
import torch
import importlib
import pandas as pd

from datetime import date
from tqdm import tqdm
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForMaskedLM

from utils.generate_utils import calc_ppl



os.chidr("/home/a03-sgoel/FLaT")
todays_date = date.today().strftime('%d-%m')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -------# Sequence embeds / logits #-------- #
def get_logits(tokens, model):
    with torch.no_grad():
        logits = model(**tokens).logits.squeeze(0)  # L, V
    return logits

def get_embeds(tokens, model):
    with torch.no_grad():
        embeds = model(**tokens, return_hidden_states=True).mean(dim=1)  # B=1, D
    return embeds


# -------# Model loading #-------- #
def load_model_and_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.lm.pretrained_esm)
    model = AutoModelForMaskedLM.from_pretrained(config.lm.pretrained_esm).eval().to(device)
    return tokenizer, model

def load_transport_model(mode: str, prop: str, config):
    models = importlib.import_module(f"src.latent_transport.{mode}")
    if prop == "solubility":
        guidance_model = models.SolubilityClassifier(config)
    elif prop == "stability":
        guidance_model = models.StabilityRegressor(config)
    elif prop == "permeability":
        guidance_model = models.PermeabiltyRegressor(config)

    pl_module = importlib.import_module(f"src.latent_transport.{mode}.{prop}")
    transport_model = pl_module.TransportModule(config, guidance_model)
    transport_model.load_from_checkpoint(config.checkpointing.best_ckpt_path)
    return transport_model.to(device)


# -------# Langevin transport step #-------- #
def langevin_transport(latent, transport_model, mode, eps):
    if mode == "energy":
        # TODO: double check and fix based on grads of energy(z_t) w.r.t z?
        latent = latent.detach().clone().requires_grad_(True)
        optim_latent = transport_model(latent)
        optim_latent.backward(retain_graph=True)
    elif mode == "score":
        # TODO: implement this
        ...


# -------# Jacobian decoding #-------- #
def jacobian_decoding():
    ...


# -------# Entry point for running #-------- #
def main():
    mode = "score"
    prop = "solubility"

    # TODO: add sampling params section with eps, langevin_steps, decoding_steps
    config = OmegaConf.load(f"./src/configs/{mode}/{prop}.yaml")

    tokenizer, esm_model = load_model_and_tokenizer(config)
    transport_model = load_transport_model(mode, prop, config)

    seq_data = pd.read_csv(f"./data/{mode}/{prop}/test.csv")
    seqs = seq_data['Sequence'].tolist()
    perplexities = []

    for seq in tqdm(seqs, desc="Optimizing sequences"):
        tokens = tokenizer(seq, return_tensors='pt')

        latent = get_embeds(tokens, esm_model)
        for l in range(len(config.sampling.langevin_steps)):
            # TODO: finish langevin_transport() for both energy and score cases
            transported_latent = langevin_transport(latent, transport_model, mode, config.sampling.eps)
        
        logits = get_logits(tokens, esm_model)
        for d in range(len(config.sampling.decoding_steps)):
            # TODO: implement and call jacobian_decoding()
            logits = jacobian_decoding(...)
        
        optim_seq = torch.distributions.Categorical(logits=logits).sample()

        perplexities.append(calc_ppl(esm_model, tokenizer, optim_seq))

    
    seq_data['PPL'] = perplexities
    seq_data.to_csv(f'./results/{mode}/{prop}/{todays_date}_optim_seqs.csv', index=False)


if __name__ == "__main__":
    main()
