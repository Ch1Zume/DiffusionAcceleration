import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class Policy(nn.Module):
    """
    加速策略网络，支持三种预测模式：
    
    - mode='full_pred': 同时预测 action（full/cache）和 order（Taylor 展开阶数）
    - mode='order_pred': 只预测 order，action 由外部 build_action_array 根据 interval 决定
    - mode='action_pred': 只预测 action（full/cache），order 固定为 max_order
    
    输出:
        - action: 0 = full compute, 1 = cache (full_pred 和 action_pred 模式)
        - order: Taylor 展开阶数 (0 到 max_order)
    """
    
    def __init__(self, T, num_actions=2, max_order=2, latent_channel=16, hidden_dim=32, num_layers=10, mode='full_pred'):
        """
        Args:
            T: 总步数
            num_actions: 动作数量 (full_pred 和 action_pred 模式使用)
            max_order: 最大 Taylor 展开阶数
            latent_channel: latent 通道数
            hidden_dim: MLP 隐藏层维度
            num_layers: MLP 层数
            mode: 预测模式，'full_pred'、'order_pred' 或 'action_pred'
        """
        super().__init__()
        assert mode in ['full_pred', 'order_pred', 'action_pred'], f"mode must be 'full_pred', 'order_pred' or 'action_pred', got {mode}"
        
        # 保存所有配置参数，方便 save/load
        self.T = T
        self.num_actions = num_actions
        self.max_order = max_order
        self.num_orders = max_order + 1
        self.latent_channel = latent_channel
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.mode = mode
        
        # 强制 full compute 的边界步数
        self.force_full_start = 1      # 前 N 步强制 full
        self.force_full_end = 2        # 后 N 步强制 full

        # Latent encoder: [B, C, H, W] -> [B, C]
        self.latent_encoder = nn.Sequential(
            nn.Conv2d(latent_channel, latent_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        # MLP: 输入维度 = latent_channel + t_norm + distance + interval
        in_dim = latent_channel + 3
        layers = []
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)

        # 输出头
        if mode in ['full_pred', 'action_pred']:
            self.action_head = nn.Linear(hidden_dim, num_actions)
        if mode in ['full_pred', 'order_pred']:
            self.order_head = nn.Linear(hidden_dim, self.num_orders)

        self._init_weights()

    def _init_weights(self):
        """正交初始化所有线性层"""
        gain = (2.0) ** 0.5
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)

    def forward(self, t_current, latent_current, distance, interval, sft=False):
        """
        Args:
            t_current: 当前 timestep, [B]
            latent_current: 当前 latent, [B, C, H, W]
            distance: 距离上次 full compute 的步数, [B]
            interval: 当前 full compute 的间隔, [B]
            sft: 是否返回 logits (用于 SFT 训练)
        
        Returns:
            mode='full_pred':
                sft=True:  (logits_action, logits_order)
                sft=False: (probs_action, probs_order)
            mode='order_pred':
                sft=True:  logits_order
                sft=False: probs_order
            mode='action_pred':
                sft=True:  logits_action
                sft=False: probs_action
        """
        B, C = latent_current.shape[:2]
        
        # Encode latent
        latent = self.latent_encoder(latent_current.to(self.latent_encoder[0].weight.dtype))
        latent = latent.view(B, C)
        
        # 归一化 timestep
        t_norm = t_current.float().unsqueeze(-1) / max(self.T - 1, 1)
        distance = distance.float().unsqueeze(-1)
        interval = interval.float().unsqueeze(-1)
        
        # 拼接特征
        x = torch.cat([t_norm, distance, interval, latent], dim=1)
        h = self.mlp(x)
        
        if self.mode == 'full_pred':
            logits_action = self.action_head(h)
            logits_order = self.order_head(h)
            if sft:
                return logits_action, logits_order
            return F.softmax(logits_action, dim=-1), F.softmax(logits_order, dim=-1)
        elif self.mode == 'order_pred':
            logits_order = self.order_head(h)
            if sft:
                return logits_order
            return F.softmax(logits_order, dim=-1)
        else:  # action_pred
            logits_action = self.action_head(h)
            if sft:
                return logits_action
            return F.softmax(logits_action, dim=-1)

    def _is_forced_full(self, t_idx, total_steps):
        """判断当前步是否强制 full compute"""
        return t_idx < self.force_full_start or t_idx >= total_steps - self.force_full_end

    # ====================== full_pred 模式的方法 ======================
    
    def sample_action(self, actions, t_current, t_idx, latent_current, distance, interval, batch_wise=True, sft=False):
        """
        采样动作和阶数 (仅 full_pred 模式)。
        
        Args:
            actions: 动作字典 {'full': 0, 'cache': 1}
            t_current: 当前 timestep, [B]
            t_idx: 当前步索引
            latent_current: 当前 latent, [B, C, H, W]
            distance: 距离上次 full compute 的步数, [B]
            interval: 上次 full compute 的间隔, [B]
            batch_wise: 是否对整个 batch 采样同一动作
            sft: 是否为 SFT 模式
        
        Returns:
            action: 采样的动作, [B]
            order: 采样的阶数, [B]
            log_probs: 对数概率 log π(a, k | s), [B]
        """
        assert self.mode == 'full_pred', "sample_action is only available in 'full_pred' mode"
        
        B = t_current.shape[0]
        device = t_current.device

        # 强制 full compute 的边界步
        if self._is_forced_full(t_idx, self.T):
            action = torch.full((B,), actions['full'], device=device, dtype=torch.long)
            order = torch.full((B,), self.max_order, device=device, dtype=torch.long)
            # 返回可反传的零对数概率
            log_probs = t_current.float().sum() * 0.0  # 保持计算图连接
            log_probs = log_probs.expand(B)
            return action, order, log_probs

        # 获取概率分布
        out = self.forward(t_current, latent_current, distance, interval, sft=sft)
        if sft:
            probs_action = F.softmax(out[0], dim=-1)
            probs_order = F.softmax(out[1], dim=-1)
        else:
            probs_action, probs_order = out

        if batch_wise:
            return self._sample_action_batch_wise(actions, probs_action, probs_order, B, device)
        else:
            return self._sample_action_element_wise(actions, probs_action, probs_order, B, device)

    def _sample_action_batch_wise(self, actions, probs_action, probs_order, B, device):
        """Batch 内所有样本采样相同动作"""
        # 聚合概率并采样动作
        probs_action_agg = probs_action.mean(dim=0)
        dist_a = torch.distributions.Categorical(probs=probs_action_agg)
        a = dist_a.sample()
        log_probs = dist_a.log_prob(a).expand(B)
        action = a.expand(B).long()

        # Full compute 时使用 max_order
        if a.item() == actions['full']:
            order = torch.full((B,), self.max_order, device=device, dtype=torch.long)
            return action, order, log_probs

        # Cache 时采样 order
        probs_order_agg = probs_order.mean(dim=0)
        dist_k = torch.distributions.Categorical(probs=probs_order_agg)
        k = dist_k.sample()
        log_probs = log_probs + dist_k.log_prob(k).expand(B)
        order = k.expand(B).long()
        
        return action, order, log_probs

    def _sample_action_element_wise(self, actions, probs_action, probs_order, B, device):
        """每个样本独立采样动作"""
        dist_a = torch.distributions.Categorical(probs=probs_action)
        action = dist_a.sample().long()
        log_probs = dist_a.log_prob(action)

        is_full = (action == actions['full'])
        order = torch.full((B,), self.max_order, device=device, dtype=torch.long)

        # 只对 cache 的样本采样 order
        if (~is_full).any():
            dist_k = torch.distributions.Categorical(probs=probs_order)
            k_all = dist_k.sample().long()
            logp_k_all = dist_k.log_prob(k_all)
            
            order = torch.where(is_full, order, k_all)
            log_probs = log_probs + torch.where(is_full, torch.zeros_like(logp_k_all), logp_k_all)

        return action, order, log_probs

    # ====================== order_pred 模式的方法 ======================
    
    def sample_order(self, action, actions, t_current, t_idx, latent_current, distance, interval, batch_wise=True, sft=False):
        """
        根据给定的 action 采样 order (仅 order_pred 模式)。
        
        注意：action 由外部根据 build_action_array(interval, T) 决定，
        本方法只负责在 cache 步时采样 order。
        
        Args:
            action: 当前步的动作 (0=full, 1=cache)，标量 int
            actions: 动作字典 {'full': 0, 'cache': 1}
            t_current: 当前 timestep, [B]
            t_idx: 当前步索引
            latent_current: 当前 latent, [B, C, H, W]
            distance: 距离上次 full compute 的步数, [B]
            interval: 当前 full compute 的间隔, [B]
            batch_wise: 是否对整个 batch 采样同一 order
            sft: 是否为 SFT 模式
        
        Returns:
            order: 采样的阶数, [B]
            log_probs: 对数概率 log π(k | s), [B]
        """
        assert self.mode == 'order_pred', "sample_order is only available in 'order_pred' mode"
        
        B = t_current.shape[0]
        device = t_current.device

        # Full compute 步使用 max_order，无需采样
        if action == actions['full']:
            order = torch.full((B,), self.max_order, device=device, dtype=torch.long)
            # 返回可反传的零对数概率
            log_probs = t_current.float().sum() * 0.0  # 保持计算图连接
            log_probs = log_probs.expand(B)
            return order, log_probs

        # Cache 步：采样 order
        out = self.forward(t_current, latent_current, distance, interval, sft=sft)
        if sft:
            probs_order = F.softmax(out, dim=-1)
        else:
            probs_order = out

        if batch_wise:
            return self._sample_order_batch_wise(probs_order, B, device)
        else:
            return self._sample_order_element_wise(probs_order, B, device)

    def _sample_order_batch_wise(self, probs_order, B, device):
        """Batch 内所有样本采样相同 order"""
        probs_order_agg = probs_order.mean(dim=0)
        dist_k = torch.distributions.Categorical(probs=probs_order_agg)
        k = dist_k.sample()
        log_probs = dist_k.log_prob(k).expand(B)
        order = k.expand(B).long()
        
        return order, log_probs

    def _sample_order_element_wise(self, probs_order, B, device):
        """每个样本独立采样 order"""
        dist_k = torch.distributions.Categorical(probs=probs_order)
        order = dist_k.sample().long()
        log_probs = dist_k.log_prob(order)

        return order, log_probs

    # ====================== action_pred 模式的方法 ======================
    
    def sample_action_only(self, actions, t_current, t_idx, latent_current, distance, interval, batch_wise=True, sft=False):
        """
        只采样 action，order 固定为 max_order (仅 action_pred 模式)。
        
        Args:
            actions: 动作字典 {'full': 0, 'cache': 1}
            t_current: 当前 timestep, [B]
            t_idx: 当前步索引
            latent_current: 当前 latent, [B, C, H, W]
            distance: 距离上次 full compute 的步数, [B]
            interval: 上次 full compute 的间隔, [B]
            batch_wise: 是否对整个 batch 采样同一动作
            sft: 是否为 SFT 模式
        
        Returns:
            action: 采样的动作, [B]
            order: 固定为 max_order, [B]
            log_probs: 对数概率 log π(a | s), [B]
        """
        assert self.mode == 'action_pred', "sample_action_only is only available in 'action_pred' mode"
        
        B = t_current.shape[0]
        device = t_current.device

        # 强制 full compute 的边界步
        if self._is_forced_full(t_idx, self.T):
            action = torch.full((B,), actions['full'], device=device, dtype=torch.long)
            order = torch.full((B,), self.max_order, device=device, dtype=torch.long)
            # 返回可反传的零对数概率
            log_probs = t_current.float().sum() * 0.0  # 保持计算图连接
            log_probs = log_probs.expand(B)
            return action, order, log_probs

        # 获取 action 概率分布
        out = self.forward(t_current, latent_current, distance, interval, sft=sft)
        if sft:
            probs_action = F.softmax(out, dim=-1)
        else:
            probs_action = out

        # order 固定为 max_order
        order = torch.full((B,), self.max_order, device=device, dtype=torch.long)

        if batch_wise:
            return self._sample_action_only_batch_wise(probs_action, order, B, device)
        else:
            return self._sample_action_only_element_wise(probs_action, order, B, device)

    def _sample_action_only_batch_wise(self, probs_action, order, B, device):
        """Batch 内所有样本采样相同 action"""
        probs_action_agg = probs_action.mean(dim=0)
        dist_a = torch.distributions.Categorical(probs=probs_action_agg)
        a = dist_a.sample()
        log_probs = dist_a.log_prob(a).expand(B)
        action = a.expand(B).long()

        return action, order, log_probs

    def _sample_action_only_element_wise(self, probs_action, order, B, device):
        """每个样本独立采样 action"""
        dist_a = torch.distributions.Categorical(probs=probs_action)
        action = dist_a.sample().long()
        log_probs = dist_a.log_prob(action)

        return action, order, log_probs


def save_policy(policy: nn.Module, save_path: str):
    """保存策略网络到文件"""
    if hasattr(policy, "module"):
        policy = policy.module

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "state_dict": policy.state_dict(),
        "config": {
            "T": policy.T,
            "num_actions": policy.num_actions,
            "max_order": policy.max_order,
            "latent_channel": policy.latent_channel,
            "hidden_dim": policy.hidden_dim,
            "num_layers": policy.num_layers,
            "mode": policy.mode,
        },
    }
    torch.save(checkpoint, save_path)


def load_policy(load_path: str, device="cpu", strict=True):
    """从文件加载策略网络"""
    checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]

    # 兼容旧 checkpoint（保存 num_orders）和新格式（保存 max_order）
    max_order = cfg.get("max_order", cfg.get("num_orders", 3) - 1)
    
    # 兼容旧 checkpoint（从模型结构推断参数）
    num_actions = cfg.get("num_actions", 2)
    latent_channel = cfg.get("latent_channel", 16)
    hidden_dim = cfg.get("hidden_dim", 32)
    num_layers = cfg.get("num_layers", 10)
    mode = cfg.get("mode", "full_pred")  # 默认兼容旧版本

    policy = Policy(
        T=cfg["T"],
        num_actions=num_actions,
        max_order=max_order,
        latent_channel=latent_channel,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        mode=mode,
    )

    policy.load_state_dict(checkpoint["state_dict"], strict=strict)
    policy.to(device)
    return policy
