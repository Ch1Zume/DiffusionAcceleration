import torch
from torch.distributions.utils import probs_to_logits
import torch.nn as nn
import torch.distributions as D
import math
import torch.nn.functional as F
from pathlib import Path

# TODO: Incorporate more inputs
class Policy(nn.Module):
    def __init__(self, T, num_actions=2, latent_channel=16, hidden_dim=32, num_layers=10):
        super().__init__()
        self.T = T

         # 1) 卷积 encoder：把 [B,C,H,W] 压成 [B,C]
        self.latent_encoder = nn.Sequential(
            nn.Conv2d(latent_channel, latent_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # -> [B,C,1,1]
        )

        in_dim_mlp = latent_channel + 1 + 1 + 1
        # in_dim_mlp = 1
        layers = []

        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim_mlp, hidden_dim))
            layers.append(nn.SiLU())
            in_dim_mlp = hidden_dim

        # output layer
        layers.append(nn.Linear(in_dim_mlp, num_actions))
        self.mlp = nn.Sequential(*layers)

        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=torch.sqrt(torch.tensor(2.0)))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, t_current, latent_current, distance, interval, batch_wise, sft):
        """
        Input:
            t_current: current sampling step, [batch_size, ]
            latent_current: current image latents
            cache_history: timesteps to last caching
        Return:
            probs_full: probability of selecting full compute at current sampling step, [batch_size, ]

        Basic policy:
        1. Closer to the end of sampling, more caching (in progress)
        2. Higher similarity between latents of adjacent timesteps, more caching (in progress)
        3. Larger distance, more full compute;  
        """
        B = latent_current.shape[0]
        C =  latent_current.shape[1]
        # 确保输入与卷积权重同 dtype，避免 fp16 输入/float bias 冲突
        latent = self.latent_encoder(latent_current.to(self.latent_encoder[0].weight.dtype))
        latent = latent.view(B, C)
        t = t_current.unsqueeze(-1).float()
        distance = distance.unsqueeze(-1).float()
        interval = interval.unsqueeze(-1).float()

        if sft:
            latent = latent * 1.0
            distance = distance * 1.0
            interval = interval * 1.0
        # 归一化的目的：不同sampler的t排布不一样，归一化是为了了解相对位置
        t_norm = t / float(self.T - 1)

        x = torch.cat([t_norm, distance, interval, latent], dim=1) # [B, N+3]
        # x = t_norm
        if batch_wise:
            # aggregate features to eliminate batch dimension
            x_agg = x.mean(dim=0, keepdim=True)
            logits = self.mlp(x_agg).squeeze(-1)  # [num_actions]
            # print(logits)
        else:
            logits = self.mlp(x).squeeze(-1)  # [B, num_actions]

        if sft:
            return logits
        else:
            probs = F.softmax(logits, dim=-1)  # [B, num_actions]
            return probs

    def sample_action(self, actions, t_current, t_idx, latent_current, distance, interval, batch_wise, sft):
        """
        return:
            actions: 0 = reuse cache, 1 = full, [batch_size, ]
            log_probs: log π(a_t|t), [batch_size, ]
        """
        B = t_current.shape[0]

        # Force the initial 6 steps to be full compute, same as that in TaylorSeer
        if t_idx < 7:
            action = torch.full((B,), actions['full'], device=t_current.device, dtype=torch.float32)
            # 仍然返回可反传的零对数概率，避免后续 backward 报 requires_grad=False
            log_probs = torch.zeros(B, device=t_current.device, requires_grad=True)
            return action, log_probs

        # probability of selecting different actions at current sampling step
        probs = self.forward(t_current, latent_current, distance, interval, batch_wise, sft)      # [B, num_actions]
        # This is a sampling machine
        dist = torch.distributions.Categorical(probs=probs)
        # randomly select actions, 采样出来的是动作对应的那个编号，自己定义哪个是哪个
        action1 = dist.sample()                   # [1]
        log_probs1 = dist.log_prob(action1)        # [1]

        action  = action1.expand(B)    # [B]
        log_probs = log_probs1.expand(B)      # [B]

        return action, log_probs

def save_policy(policy: nn.Module, save_path: str):
    """
    Save policy network.

    Args:
        policy: Policy model
        save_path: path to save, e.g. "ckpts/policy.pt"
    """
    if hasattr(policy, "module"):
        policy = policy.module
        
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "state_dict": policy.state_dict(),
        "config": {
            "T": policy.T,
            "num_actions": policy.mlp[-1].out_features,
            "latent_channel": policy.latent_encoder[0].in_channels,
            "hidden_dim": policy.mlp[0].out_features,
            "num_layers": len(
                [m for m in policy.mlp if isinstance(m, torch.nn.Linear)]
            ),
        },
    }

    torch.save(checkpoint, save_path)

def load_policy(load_path: str, device="cpu", strict=True):
    """
    Load policy network.

    Args:
        load_path: checkpoint path
        device: target device
        strict: whether to strictly load state_dict

    Returns:
        policy: loaded Policy model
    """
    checkpoint = torch.load(load_path, map_location="cpu")

    cfg = checkpoint["config"]

    policy = Policy(
        T=cfg["T"],
        num_actions=cfg["num_actions"],
        latent_channel=cfg["latent_channel"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
    )

    policy.load_state_dict(checkpoint["state_dict"], strict=strict)
    policy.to(device)

    return policy