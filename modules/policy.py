import torch
from torch.distributions.utils import probs_to_logits
import torch.nn as nn
import torch.distributions as D
import math
import torch.nn.functional as F

# TODO: Incorporate more inputs
class Policy(nn.Module):
    def __init__(self, T, num_actions=2, latent_channel=16, hidden_dim=32, num_layers=3, init_p=0.8):
        super().__init__()
        self.T = T

         # 1) 卷积 encoder：把 [B,C,H,W] 压成 [B,C]
        self.latent_encoder = nn.Sequential(
            nn.Conv2d(latent_channel, latent_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # -> [B,C,1,1]
        )

        in_dim_mlp = latent_channel + 1 + 1
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

        # 2. 设置最后一层 bias，使得初始 p_full = init_p_full
        logit = math.log(init_p / (1.0 - init_p))
        last_linear = self.mlp[-1]
        last_linear.bias.data.fill_(logit)
        # 这样一开始 logits ≈ logit，probs_full = sigmoid(logit) ≈ init_p_full

    def forward(self, t_current, latent_current, distance):
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
        latent = self.latent_encoder(latent_current)
        latent = latent.view(B, C)
        t = t_current.unsqueeze(-1).float()
        distance = distance.unsqueeze(-1).float()
        # 归一化的目的：不同sampler的t排布不一样，归一化是为了了解相对位置
        t_norm = t / float(self.T - 1)
        # print(t_norm.shape)
        # print(distance.shape)
        # print(latent.shape)

        x = torch.cat([t_norm, distance, latent], dim=1) # [B, N+2]

        logits = self.mlp(x).squeeze(-1)  # [B, num_actions]
        probs = F.softmax(logits)  # [B, num_actions]
        return probs

    def sample_action(self, actions, t_current, latent_current, distance):
        """
        return:
            actions: 0 = reuse cache, 1 = full, [batch_size, ]
            log_probs: log π(a_t|t), [batch_size, ]
        """
        # Force the initial 3 steps to be full compute
        if torch.all(t_current < 3):
            B = t_current.shape[0]
            action = torch.full((B,), actions['full'], device=t_current.device, dtype=torch.float32)
            # this force is irrelevant to the policy
            log_probs = torch.zeros(B, device=t_current.device)
            return action, log_probs

        # probability of selecting different actions at current sampling step
        probs = self.forward(t_current, latent_current, distance)      # [B, num_actions]
        # This is a sampling machine
        dist = torch.distributions.Categorical(probs=probs)
        # randomly select actions, 采样出来的是动作对应的那个编号，自己定义哪个是哪个
        action = dist.sample()                   # [B]
        log_probs = dist.log_prob(action)        # [B]

        # no more than 3 consecutive caching
        mask_force_full = distance <= -3     # [B] bool
        if mask_force_full.any():
            action = action.clone()
            log_probs = log_probs.clone()
            action[mask_force_full] = actions['full']
            log_probs[mask_force_full] = 0.0
        return action, log_probs