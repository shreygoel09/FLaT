#!/usr/bin/env python3

import os
import sys
import torch
import math
import pandas as pd
import torch.nn.functional as F

from datetime import date
from tqdm import tqdm
from omegaconf import OmegaConf

from utils.generate_utils import (
    calc_ppl, calc_entropy, calc_hamming, calc_property_val,
    gen_random_tokens, get_logits, get_embeds, load_model_and_tokenizer, load_transport_model
)
from utils.model_utils import _print


os.chdir("/home/a03-sgoel/FLaT")
todays_date = date.today().strftime('%d-%m')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


mode = "energy"  # energy / score
prop = "solubility"  # solubility / permeability / stability


# -------# Langevin Transport #-------- #
def langevin_transport(batch, transport_model, mode, eta, eps):
    """ Function to perform 1 step of latent Langevin transport """
    z_t, mask = batch['latent'], batch['attention_mask']
    noise = math.sqrt(2 * eps) * torch.randn_like(z_t, device=z_t.device)
    z_t = z_t.clone().detach().requires_grad_(True)
    
    if mode == "energy":
        energy_vector = transport_model(batch={'embeds': z_t, 'attention_mask': mask})
        _print(f'energy: {energy_vector}')
        energy_vector.squeeze().backward()  # energy is already scalar
        z_t = z_t - (eta * z_t.grad) #+ noise
    
    elif mode == "score":
        with torch.no_grad():
            s_theta, _, _ = transport_model({"embeds": z_t})
            _print(f'score norm: {s_theta.norm(p=2, dim=1)}')
        _print(f'score: {s_theta}')
        z_t = z_t + (eta * s_theta) + noise

    # Update dictionary for next step
    batch['latent'] = z_t
    _print(f'zt: {z_t}')
    return z_t.detach()


# -------# Jacobian Decoding #-------- #
def jacobian_decoding(x_k, classifier, x_0, z_T, z_0, mask, encoder_lm, eta, gamma, tau):
    x_k = x_k.clone().requires_grad_(True)

    # Project logits to latent space with inverse decoder weight
    W_T = encoder_lm.lm_head.decoder.weight.T  # D, V
    W_pinv = torch.linalg.pinv(W_T)  # V, D
    embeds = x_k @ W_pinv  # L, D (Remove BOS / EOS)
    _print(f'embed decoder proj: {embeds}')
    
    # _print(f'xk shape: {x_k.shape}')
    # x_soft = F.softmax(x_k / tau, dim=-1)  # [L, V]
    # embedding_matrix = encoder_lm.get_input_embeddings().weight  # [V, D]
    # embeds = x_soft @ embedding_matrix  # [L, D]
    # _print(f'embeds soft: {embeds}')

    # Residual to use in vector-Jacobian product of relaxed logits wrt latent
    # (d_pi(x) / dx) * (pi(x)) - z')
    pi_x = embeds[1:-1, :] # Remove bos/eos dimensions
    sim = (F.cosine_similarity(z_0.mean(dim=-1), z_T.mean(dim=-1)) + 1) / 2
    z_T = sim * z_T + (1 - sim) * z_0
    residual = pi_x - z_T
    _print(f'residual: {residual}')
    latent_loss = 0.5 * torch.sum(residual ** 2) / residual.numel()  # Normalize
    _print(f'latent loss: {latent_loss}')

    # # Classifier gradients
    pred = classifier(batch={'embeds': pi_x.unsqueeze(0), 'attention_mask': mask})
    _print(f'pred: {pred}')
    if mode == "energy":
        if prop == "solubility":
            #value_loss = -pred
            value_loss = F.binary_cross_entropy_with_logits(pred, torch.ones_like(pred))
        else:
            value_loss = -pred
    elif mode == "score":
        z_t, _, _ = pred
        value_loss = F.mse_loss(z_t, z_T)
    _print(f'value loss: {value_loss}')

    kl_div = F.kl_div(
        input=F.log_softmax(x_0.detach(), dim=-1),
        target=F.log_softmax(x_k, dim=-1),
        log_target=True,
        reduction='batchmean'
    )
    _print(f'kl div: {kl_div}')

    # Update step
    #grad = latent_grad
    #grad = latent_grad + kl_grad
    #grad = latent_grad + value_grad

    #loss = latent_loss + value_loss - kl_div
    #loss = latent_loss + value_loss + kl_div # Latent loss is already negated
    loss = latent_loss
    grad = torch.autograd.grad(loss, x_k)[0].clamp(min=-1, max=1)
    _print(f'grad: {grad}')

    x_k_prime = x_k - (eta * grad)
    return x_k_prime



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
    hammings = []

    for seq in tqdm(seqs, desc="Optimizing sequences"):

        mask = torch.ones(1, len(seq)).to(device)
        
        # Perform langevin transport on the initial vector of gaussian noise
        batch = {
            'latent': torch.randn(1, len(seq), config.model.d_model).to(device),
            'attention_mask': mask
        }
        langevin_lr = torch.tensor(config.sampling.langevin.lr, device=device, dtype=torch.float32)
        langevin_eps = torch.tensor(config.sampling.langevin.noise_eps, device=device, dtype=torch.float32)
        for _ in range(config.sampling.langevin.steps):
            latent = langevin_transport(
                batch=batch,
                transport_model=transport_model,
                mode=mode,
                eta=langevin_lr,
                eps=langevin_eps
            )
        
        og_val = transport_model(batch={"embeds": latent, 'attention_mask': mask})
        _print(f'pred of z_T: {torch.sigmoid(og_val)}')
        
        # Compute and optimize sequence logits
        og_latent = get_embeds(tokenizer(seq, return_tensors='pt'), encoder_model).squeeze(0)[1:-1, :]
        logits = get_logits(tokenizer(seq, return_tensors='pt'), encoder_lm_model)
        og_logits = logits.detach().clone()
        decoder_lr = torch.tensor(config.sampling.decoding.lr, device=device, dtype=torch.float32)
        for _ in range(config.sampling.decoding.steps):
            logits = jacobian_decoding(
                x_k=logits,
                classifier=transport_model,
                x_0=og_logits,
                z_T=latent,
                z_0=og_latent,
                mask=mask,
                encoder_lm=encoder_lm_model,
                eta=decoder_lr,
                gamma=config.sampling.decoding.gamma,
                tau=(1/math.log(len(seq))),
            )

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

        _print(f'ppls:       {perplexities[-1]}')
        _print(f'og vals:    {og_vals[-1]}')
        _print(f'optim vals: {optim_vals[-1]}')
        _print(f'entropy:    {entropies[-1]}')
        _print('\n')

    # Save results
    seq_data['PPL'] = perplexities
    seq_data['OG Vals'] = og_vals
    seq_data['Optim Vals'] = optim_vals
    seq_data['entropy'] = entropies
    
    save_path = f'./results/optim/{mode}/{prop}/{todays_date}/transformer_optim_seqs.csv'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    seq_data.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()





# # -------# Jacobian Decoding #-------- #
# def jacobian_decoding(x_k, classifier, optim_latent, encoder_lm, eta, gamma, tau=1.0):
#     """
#     Function to perform 1 step of vector-jacobian product decoding.
    
#     Args:
#         - x_k (torch.Tensor): sequence logits [L, V]
#         - z_target (torch.Tensor): langevin-transported latent (D)
#         - encoder (PreTrainedModel): pre-trained encoder model. Must have get_input_embeddings() fn
#         - eta: learning rate / update step size
#     Return:
#         - x_prime (torch.Tensor): optimzed  
#     """
#     x_k = x_k.detach().clone().requires_grad_(True)
#     x_flat = x_k.view(-1)
#     _print(f'x_k: {x_k}')
#     _print(f'x flat: {x_flat}')

#     # Create embeddings from logits
#     def relax_fn(x_flat):
#         # Gumbel-softmax relaxation
#         x = x_flat.view(x_k.shape) # L, V
#         gumbel_noise = -torch.log(-torch.log(torch.rand_like(x) + 1e-8) + 1e-8)
#         x = (x + gumbel_noise) / tau

#         W_T = encoder_lm.lm_head.decoder.weight.T  # D, V
#         W_pinv = torch.linalg.pinv(W_T)  # V, D
#         embeds = x @ W_pinv  # L, D
#         return embeds
#         #return embeds.mean(dim=0).unsqueeze(0)  # 1, D

#         # probs = x.softmax(dim=-1).unsqueeze(0) # 1, L, V
#         # embed_weight = encoder.get_input_embeddings().weight
#         # word_embeds = torch.matmul(probs, embed_weight) # 1, L, D
#         # embeds = encoder(inputs_embeds=word_embeds, output_hidden_states=True).hidden_states[-1]
#         # return embeds.sum(dim=1)  # 1, D

#     # Sequence relaxation
#     pi_x = relax_fn(x_flat)[1:-1, :] # Remove BOS / EOS positions
#     _print(f'embeds: {pi_x}, {pi_x.shape}')
    
#     # Compute residual vector
#     residual = pi_x - optim_latent

#     # Vector-Jacobian product of relaxed logits wrt latent, (d_pi(x) / dx) * (pi(x)) - z')
#     latent_grad = torch.autograd.grad(
#         outputs=pi_x,
#         inputs=x_flat,
#         grad_outputs=residual, # Vector is the residuals
#         retain_graph=True
#     )[0]

#     # Classifier gradients
#     pred = classifier({'embeds': pi_x})
#     if mode == "energy":
#         value_loss = -F.binary_cross_entropy_with_logits(pred, torch.ones_like(pred))
#         if prop == "stability":
#             value_loss = value_loss
#     elif mode == "score":
#         curr_latent, _, _ = pred
#         value_loss = F.mse_loss(curr_latent, optim_latent)
#         _print(f'value loss: {value_loss}')
#     value_grad =  torch.autograd.grad(value_loss, x_flat, retain_graph=True)[0]

#     # Update step
#     grad = latent_grad + value_grad
#     _print(f'latent grad: {latent_grad}')
#     _print(f'value grad: {value_grad}')
#     _print(f'total grad: {grad}')
#     x_k_prime = x_flat - (eta * latent_grad)
#     logits = x_k_prime.view(x_k.shape).detach() # L, V
#     return logits



# # -------# Jacobian Decoding #-------- #
# def jacobian_decoding(x_k, og_embeds, optim_latent, encoder_lm, eta, tau=1.0):
#     """
#     Function to perform 1 step of vector-jacobian product decoding.
    
#     Args:
#         - x_k (torch.Tensor): sequence logits [L, V]
#         - z_target (torch.Tensor): langevin-transported latent (D)
#         - encoder (PreTrainedModel): pre-trained encoder model. Must have get_input_embeddings() fn
#         - eta: learning rate / update step size
#     Return:
#         - x_prime (torch.Tensor): optimzed  
#     """
#     x_k = x_k.detach().clone().requires_grad_(True)
#     x_flat = x_k.view(-1)

#     # Create embeddings from logits
#     def relax_fn(x_flat):
#         # Gumbel-softmax relaxation
#         x = x_flat.view(x_k.shape) # L, V
#         gumbel_noise = -torch.log(-torch.log(torch.rand_like(x) + 1e-8) + 1e-8)
#         x = (x + gumbel_noise) / tau

#         W_T = encoder_lm.lm_head.decoder.weight.T  # D, V
#         W_pinv = torch.linalg.pinv(W_T)  # V, D
#         embeds = (x @ W_pinv).unsqueeze(0)  # 1, L, D
#         return embeds, embeds.mean(dim=0)  # 1, D

#     # Sequence relaxation
#     pi_x_full, pi_x_avg = relax_fn(x_flat)
    
#     # Compute residual vector
#     residual = pi_x_avg - optim_latent

#     # Vector-Jacobian product of relaxed logits wrt latent, (d_pi(x) / dx) * (pi(x)) - z')
#     latent_grad = torch.autograd.grad(
#         outputs=pi_x_avg,
#         inputs=x_flat,
#         grad_outputs=residual, # Vector is the residuals
#         retain_graph=True
#     )[0]

#     # Compute MMD term
#     def compute_mmd(X, Y):
#         def rbf_kernel(a, b, gamma=1.0, d_model=1280):
#             a = a.unsqueeze(1)  # L, 1, D
#             b = b.unsqueeze(0)  # 1, L, D
#             distance = ((a-b) ** 2).sum(-1)  # L, L
#             return torch.exp(-(gamma / d_model) * distance)
        
#         k_XX = rbf_kernel(X, X)
#         k_XY = rbf_kernel(X, Y)
#         k_YY = rbf_kernel(Y, Y)
        
#         _print(f'k_XX: {k_XX}, {k_XX.shape}')
#         _print(f'k_YY: {k_YY}, {k_YY.shape}')
        
#         L = X.shape[0]
#         mmd = ((k_XX.sum() - k_XX.trace()) / (L * (L-1)) ) \
#             - (2 * k_XY.mean()) \
#             + ((k_YY.sum() - k_YY.trace()) / (L * (L-1)) )
#         return mmd
    
#     _print(f'pi x full: {pi_x_full}')
#     _print(f'og embeds: {og_embeds}')
#     mmd = compute_mmd(pi_x_full.squeeze(0), og_embeds.squeeze(0))
#     mmd_grad = torch.autograd.grad(mmd, x_flat, retain_graph=True)[0]
#     _print(f'mmd: {mmd}')
#     _print(f'mmd grads: {mmd_grad}')

#     # Update step
#     grad = latent_grad + mmd_grad
#     _print(f'tot grad: {grad}')
#     x_k_prime = x_flat - (eta * grad)
#     logits = x_k_prime.view(x_k.shape).detach() # L, V
#     return logits


# # -------# Jacobian Decoding #-------- #
# def jacobian_decoding(x_k, og_embeds, optim_latent, encoder, eta, gamma, tau=1.0):
#     """
#     Function to perform 1 step of vector-jacobian product decoding.
    
#     Args:
#         - x_k (torch.Tensor): sequence logits [L, V]
#         - z_target (torch.Tensor): langevin-transported latent (D)
#         - encoder (PreTrainedModel): pre-trained encoder model. Must have get_input_embeddings() fn
#         - eta: learning rate / update step size
#     Return:
#         - x_prime (torch.Tensor): optimzed  
#     """
#     x_k = x_k.detach().clone().requires_grad_(True)
#     x_flat = x_k.view(-1)

#     # Create embeddings from logits
#     def relax_fn(x_flat):
#         # Gumbel-softmax relaxation
#         x = x_flat.view(x_k.shape) # L, V
#         gumbel_noise = -torch.log(-torch.log(torch.rand_like(x) + 1e-8) + 1e-8)
#         x = (x + gumbel_noise) / tau
#         probs = x.softmax(dim=-1).unsqueeze(0) # 1, L, V
        
#         embed_weight = encoder.get_input_embeddings().weight
#         word_embeds = torch.matmul(probs, embed_weight) # 1, L, D
        
#         #word_embeds = word_embeds * (1 - (0.15 * 0.8))  # bypass EsmEmbeddings token dropout logic
#         embeds = encoder(inputs_embeds=word_embeds, output_hidden_states=True).hidden_states[-1]
#         return embeds, embeds.sum(dim=1) # ([D], [1, D])

#     # Sequence relaxation
#     pi_x_full, pi_x_avg = relax_fn(x_flat)
    
#     # Compute residual vector
#     residual = pi_x_avg - optim_latent

#     # Vector-Jacobian product of relaxed logits wrt latent, (d_pi(x) / dx) * (pi(x)) - z')
#     latent_grad = torch.autograd.grad(
#         outputs=pi_x_avg,
#         inputs=x_flat,
#         grad_outputs=residual, # Vector is the residuals
#         retain_graph=True
#     )[0]

    # # Compute MMD term
    # def compute_mmd(X, Y):
    #     def rbf_kernel(a, b, gamma=1.0, d_model=1280):
    #         a = a.unsqueeze(1)  # L, 1, D
    #         b = b.unsqueeze(0)  # 1, L, D
    #         distance = ((a-b) ** 2).sum(-1)  # L, L
    #         return torch.exp(-(gamma / d_model) * distance)
        
    #     k_XX = rbf_kernel(X, X)
    #     k_XY = rbf_kernel(X, Y)
    #     k_YY = rbf_kernel(Y, Y)
        
    #     _print(f'k_XX: {k_XX}, {k_XX.shape}')
    #     _print(f'k_YY: {k_YY}, {k_YY.shape}')
        
    #     L = X.shape[0]
    #     mmd = ((k_XX.sum() - k_XX.trace()) / (L * (L-1)) ) \
    #         - (2 * k_XY.mean()) \
    #         + ((k_YY.sum() - k_YY.trace()) / (L * (L-1)) )
    #     return torch.sqrt(mmd)
    
    # _print(f'pi x full: {pi_x_full}')
    # _print(f'og embeds: {og_embeds}')
    # mmd = compute_mmd(pi_x_full.squeeze(0), og_embeds.squeeze(0))
    # mmd_grad = torch.autograd.grad(mmd, x_flat, retain_graph=True)[0]
#     _print(f'mmd: {mmd}')
#     _print(f'mmd grads: {mmd_grad}')

#     # Update step
#     grad = latent_grad + (gamma * mmd_grad)
#     _print(f'tot grad: {grad}')
#     x_k_prime = x_flat - (eta * grad)
#     logits = x_k_prime.view(x_k.shape).detach() # L, V
#     return logits



# # -------# Jacobian Decoding #-------- #
# def jacobian_decoding(x_k, p_x, optim_latent, encoder, eta, gamma, tau=1.0):
#     """
#     Function to perform 1 step of vector-jacobian product decoding.
    
#     Args:
#         - x_k (torch.Tensor): sequence logits [L, V]
#         - z_target (torch.Tensor): langevin-transported latent (D)
#         - encoder (PreTrainedModel): pre-trained encoder model. Must have get_input_embeddings() fn
#         - eta: learning rate / update step size
#     Return:
#         - x_prime (torch.Tensor): optimzed  
#     """
#     x_k = x_k.detach().clone().requires_grad_(True)
#     x_flat = x_k.view(-1)

#     # Create embeddings from logits
#     def relax_fn(x_flat):
#         # Gumbel-softmax relaxation
#         x = x_flat.view(x_k.shape) # L, V
#         gumbel_noise = -torch.log(-torch.log(torch.rand_like(x) + 1e-8) + 1e-8)
#         x = (x + gumbel_noise) / tau
#         probs = x.softmax(dim=-1).unsqueeze(0) # 1, L, V
        
#         embed_weight = encoder.get_input_embeddings().weight
#         word_embeds = torch.matmul(probs, embed_weight) # 1, L, D
        
#         word_embeds = word_embeds * (1 - (0.15 * 0.8))  # bypass EsmEmbeddings token dropout logic
#         outs = encoder(inputs_embeds=word_embeds, output_hidden_states=True)
#         return outs.hidden_states[-1].sum(dim=1) # 1, D

#     # Residual vector
#     pi_x = relax_fn(x_flat)
#     residual = pi_x - optim_latent

#     # Vector-Jacobian product of relaxed logits wrt latent, (d_pi(x) / dx) * (pi(x)) - z')
#     latent_grad = torch.autograd.grad(
#         outputs=pi_x,
#         inputs=x_flat,
#         grad_outputs=residual, # Vector is the residuals
#         retain_graph=False
#     )[0]
#     _print(f'pi x: {pi_x}')
#     _print(f'residual: {residual}')
#     _print(f'latent grad: {latent_grad}')

#     # Grad of KL[optim seq | og seq] wrt optim seq
#     qx_k = x_k.log_softmax(dim=-1)
#     p_x = p_x.softmax(dim=-1)
#     kl = F.kl_div(qx_k, p_x)
#     kl_grad = gamma * torch.autograd.grad(kl, x_k)[0].view(-1)
#     _print(f'x log sm: {x_k.log_softmax(dim=-1)}')
#     _print(f'p x: {p_x}')
#     _print(f'kl: {kl}')
#     _print(f'kl grad: {kl_grad}, {kl_grad.shape}')

#     # Update step
#     grad = latent_grad + kl_grad
#     _print(f'tot grad: {grad}')
#     x_k_prime = x_flat - (eta * grad)
#     logits = x_k_prime.view(x_k.shape).detach() # L, V
#     return logits




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


