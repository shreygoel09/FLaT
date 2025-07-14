#!/usr/bin/env python3

import os
import pandas as pd

from datetime import date
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.latent_transport.score.permeability.tokenizer import SMILES_SPE_Tokenizer
from utils.generate_utils import calc_ppl
from utils.model_utils import _print


device = "cuda"
os.chdir('/home/a03-sgoel/FLaT')
todays_date = date.today().strftime('%d-%m')


prop = "stability"


# Randomly select 1k sequences for benchmarking
df = pd.read_csv(f"./data/{prop}/seqs_for_optim.csv")

df['OG Sequence PPL'] = None


if prop in {"solubility", "stability"}:
    encoder_model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
elif prop == "permeability":
    encoder_model = AutoModelForMaskedLM.from_pretrained("aaronfeller/PeptideCLM-23M-all")
    tokenizer = SMILES_SPE_Tokenizer(
        "/home/a03-sgoel/FLaT/src/latent_transport/score/permeability/new_vocab.txt",
        "/home/a03-sgoel/FLaT/src/latent_transport/score/permeability/new_splits.txt"
    )


seqs = df['Sequence'].tolist()
for seq in tqdm(seqs, desc='Calculating PPL'):
    try:
        ppl = calc_ppl(encoder_model, tokenizer, seq)
    except:
        ppl = None
    _print(f'ppl: {ppl}')
    df.loc[df['Sequence'] == seq, 'OG Sequence PPL'] = ppl


save_path = f'./results/true_data/{prop}/'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
df.to_csv(save_path + "seqs_with_ppl.csv", index=False)
