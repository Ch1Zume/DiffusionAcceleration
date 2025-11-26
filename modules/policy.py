import torch
import torch.nn as nn
import torch.distributions as D
import math

# TODO: Incorporate more inputs
class Policy(nn.Module):
    def __init__(self, T, hidden_dim=32, num_layers=3, init_p_full=0.8):
        super().__init__()
        self.T = T

        layers = []
        in_dim = 2 

        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

        # 2. 设置最后一层 bias，使得初始 p_full = init_p_full
        logit = math.log(init_p_full / (1.0 - init_p_full))
        last_linear = self.mlp[-1]
        last_linear.bias.data.fill_(logit)
        # 这样一开始 logits ≈ logit，probs_full = sigmoid(logit) ≈ init_p_full

    def forward(self, t_current, sigma_current):
        """
        Input:
            t_current: current sampling step, [batch_size, ]
        Return:
            probs_full: probability of selecting full compute at current sampling step, [batch_size, ]
        """
        t = t_current.view(-1).float()
        sigma = sigma_current.view(-1).float()

        # Normalization
        if self.T > 1:
            t_norm = t / float(self.T - 1)
        else:
            t_norm = torch.zeros_like(t)
        sigma_feat = torch.log(sigma + 1e-8)

        x = torch.stack([t_norm, sigma_feat], dim=-1)  # [B, 1]
        logits = self.mlp(x).squeeze(-1)  # [B]
        probs_full = torch.sigmoid(logits)  # [B]，0~1 概率
        return probs_full

    def sample_action(self, t_current, sigma_current):
        """
        return:
            actions: 0 = reuse cache, 1 = full, [batch_size, ]
            log_probs: log π(a_t|t), [batch_size, ]
        """
        # probability of selecting full compute at current sampling step
        probs_full = self.forward(t_current, sigma_current)      # [B]
        # prob of 1: probs_full; prob of 0: 1-probs_full
        # This is a sampling machine
        dist = D.Bernoulli(probs_full)
        # randomly select actions
        actions = dist.sample()                   # [B]
        log_probs = dist.log_prob(actions)        # [B]
        return actions, log_probs