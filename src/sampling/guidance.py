from abc import ABC, abstractmethod
from typing import Callable, Any
import torch
from torch.nn import functional as F
from torch import Tensor
from torch.distributions import Categorical
from torch.optim import Adagrad
from einops import rearrange
from reinmax import reinmax
from gdd.sampling import Denoiser, LambdaDenoiser, Planner

RewardFunction = Callable[[Tensor, dict], Tensor]
"""
Represents a reward function used in guidance. The first parameter is the
sample to evaluate, the second is the batch data for which the sample was
generated.
"""


DiffusionStepFunction = Callable[
    [dict, Denoiser, Any, float, int, int, Tensor | None, Planner | None], Tensor
]
"""
Represents a function that takes a step in a discrete diffusion process.
"""


class DiffusionGuidance(ABC):
    """
    Abstraction for a given guidance mechanism.
    """

    @abstractmethod
    def guide(
        self,
        x: dict,
        denoiser: Denoiser,
        dt: float,
        step: int,
        max_step: int,
        tokenizer,
        planner: Planner | None,
        y: Tensor | None,
        generator: DiffusionStepFunction,
        reward: RewardFunction | None,
    ) -> Tensor:
        """
        Applies the guidance method to the given generator and reward function.

        Args:
            generator: The diffusion step function.
            reward: The reward function.
            kwargs: Additional arguments for the guidance method.

        Returns:
            The guided tensor.
        """

    def requires_grad(self) -> bool:
        """
        Returns whether the guidance method requires gradients evaluation for the
        reward function.
        """
        return False


class NoGuidance(DiffusionGuidance):
    """
    No guidance method. This is the default behavior.
    """

    def guide(self, x, denoiser, dt, step, max_step, tokenizer, planner, y, generator, reward):
        return generator(x, denoiser, tokenizer, dt, step, max_step, y, planner)


class SVDD(DiffusionGuidance):
    """
    Soft Value-based Decoding in Diffusion models guidance method.
    """

    def __init__(self, alpha: float = 0.0, m: int = 10, maximize: bool = True):
        """
        Args:
            alpha: The scaling factor for the reward.
            m: The number of proto-samples to use for the guidance.
            maximize: Whether to maximize or minimize the reward.
        """
        self.alpha = alpha
        self.m = m
        self.sign = 1.0 if maximize else -1.0

    def guide(self, x, denoiser, dt, step, max_step, tokenizer, planner, y, generator, reward):
        # as there could be batch-level operations, it may be better
        # to vmap out the `m` repeated dimension, instead of the batch
        # despite the cost of rearranging the final tensor
        use_x = {}
        vmap_dims = {}
        reward = torch.vmap(reward)
        for k, v in x.items():
            if isinstance(v, Tensor):
                use_x[k] = v.expand(self.m, *v.shape)
                vmap_dims[k] = 0
            else:
                use_x[k] = v
                vmap_dims[k] = None
        vmap_generator = torch.vmap(
            generator,
            in_dims=(vmap_dims, None, None, None, None, None, None, None),
            randomness="different",
        )
        gen = vmap_generator(use_x, denoiser, tokenizer, dt, step, max_step, y, planner)
        rewards = rearrange((self.sign * reward(gen) / self.alpha), "m b -> b m", m=self.m).softmax(dim=-1)
        if self.alpha > 1e-8:
            # sample from rewards
            indices = Categorical(rewards).sample()
        else:
            # alpha is zero!
            # take the argmax
            indices = rewards.argmax(dim=-1)
        # can rearrange now
        gen = rearrange(gen, "m b ... -> b m ...")
        # shape[2:] past b and m
        final_gen = torch.gather(
            gen,
            1,
            # fmt: off
            indices.view(*((-1, 1,) + (1,) * (gen.ndim - 2))).expand(-1, 1, *gen.shape[2:]),
            # fmt: on
        ).squeeze(
            1
        )  # squeeze the m dimension
        return final_gen


class SimpleGuidance(DiffusionGuidance):
    """
    Simple guidance method, as described in https://arxiv.org/pdf/2412.10193.
    """

    def __init__(self, gamma: float = 1.0, maximize: bool = True):
        """
        Args:
            gamma: The temperature of the guidance.
        """
        self.gamma = gamma
        self.sign = 1.0 if maximize else -1.0

    def requires_grad(self):
        return True

    def guide(self, x, denoiser, dt, step, max_step, tokenizer, planner, y, generator, reward):
        def adapted_denoiser(state, step, max_step, y):
            # perform the usual step
            logits = denoiser.denoise(state, step, max_step, y)
            # but also compute the gradient of the reward given the previous xt
            xt = F.one_hot(state["output_tokens"], num_classes=logits.shape[-1]).to(torch.get_default_dtype())
            xt.requires_grad_(True)
            with torch.enable_grad() and torch.inference_mode(False):
                # same as a log probability
                reward_value = self.sign * reward(xt)
                reward_value.sum().backward()
                grad = xt.grad
            xt.requires_grad_(False)

            classifier_log_prob_ratio = (
                (grad - (xt * grad).sum(dim=-1, keepdim=True)).detach().requires_grad_(False)
            )
            classifier_log_prob = (
                (classifier_log_prob_ratio + reward_value[..., None, None]).detach().requires_grad_(False)
            )
            return logits + self.gamma * classifier_log_prob

        return generator(x, LambdaDenoiser(adapted_denoiser), tokenizer, dt, step, max_step, y, planner)


class NOS(DiffusionGuidance):
    """
    NOS guidance method, as described in https://arxiv.org/pdf/2305.20009.
    """

    def __init__(
        self,
        step_size: float = 1.0,
        stability_coeff: float = 1e-2,
        k: int = 10,
        maximize: bool = True,
        reinmax_temperature: float = 1.0,
    ):
        """
        Args:
            step_size: The step size for the guidance.
            stability_coeff: The stability coefficient for the guidance.
            k: The number of samples to use for the guidance.
        """
        self.k = k
        self.step_size = step_size
        self.stability_coeff = stability_coeff
        self.sign = -1.0 if maximize else 1.0
        self.kl = torch.nn.KLDivLoss(log_target=True)
        self.reinmax_temperature = reinmax_temperature

    def requires_grad(self):
        return True

    def guide(self, x, denoiser, dt, step, max_step, tokenizer, planner, y, generator, reward):
        # adapted from https://github.com/ngruver/NOS/blob/main/seq_models/model/mlm_diffusion.py
        # initial logits
        logits = denoiser.denoise(x, step, max_step, y)
        # optimizing the direction
        h = F.one_hot(x["output_tokens"], num_classes=logits.shape[-1]).to(logits)
        # define fix_mask for P2
        original_tokens = x["output_tokens"]
        last_mask = x["output_tokens"] == tokenizer.mask_token_id
        forbidden_pos = (x["output_tokens"] == tokenizer.eos_token_id) | (
            x["output_tokens"] == tokenizer.cls_token_id
        )
        delta = torch.zeros_like(h, requires_grad=True)
        optimizer = Adagrad([delta], lr=self.step_size)
        with torch.enable_grad() and torch.inference_mode(False):
            for _ in range(self.k):
                optimizer.zero_grad()
                h_current = h + delta
                # logits, but model handles the right way
                x["output_tokens"] = h_current
                # forward through base diffusion model
                new_logits = denoiser.denoise(x, step, max_step, y)
                # KL divergence
                kl_loss = self.kl(new_logits, logits)
                # reward loss
                updated = generator(
                    {
                        "output_tokens": reinmax(new_logits, self.reinmax_temperature)[0],
                        "last_mask": last_mask,
                        "real_tokens": original_tokens,
                        "forbidden_pos": forbidden_pos,
                    },
                    denoiser,
                    tokenizer,
                    dt,
                    step,
                    max_step,
                    planner,
                )
                reward_loss = reward(updated)
                # combine and backward
                loss = self.stability_coeff * kl_loss + self.sign * reward_loss
                loss.sum().backward()
                optimizer.step()
        final_step = denoiser.handle_non_aa_tokens(h + delta)
        ret = generator(
            {
                "output_tokens": final_step,
                "last_mask": last_mask,
                "real_tokens": original_tokens,
                "forbidden_pos": forbidden_pos,
            },
            denoiser,
            tokenizer,
            dt,
            step,
            max_step,
            planner,
        )
        return ret