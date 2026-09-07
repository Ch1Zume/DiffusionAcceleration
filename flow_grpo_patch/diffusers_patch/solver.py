import math
import torch
from diffusers.utils.torch_utils import randn_tensor
from typing import Optional, List
from dataclasses import dataclass
import torch.distributed as dist
import tqdm
from functools import partial
from collections import defaultdict
import torch.nn.functional as F
from taylorseer.ts_utils import cleanup_cache

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

def build_action_array(interval: int, length: int = 40):
    assert interval >= 1, "interval must be >= 1"

    arr = [1] * length

    # 每隔 interval 放一个 0
    for i in range(0, length, interval):
        arr[i] = 0

    # 最后两个强制为 0
    arr[-2:] = [0, 0]

    return arr



# Modified from MixGRPO
def run_sampling(
    v_pred_fn,
    z,
    timesteps,
    sigma_schedule,
    solver="flow",
    determistic=False,
    eta=0.7,
):
    assert solver in ["flow", "dance", "ddim", "dpm1", "dpm2"]
    dtype = z.dtype
    all_latents = [z]
    all_log_probs = []

    if "dpm" in solver:
        order = int(solver[-1])
        dpm_state = DPMState(order=order)
    for i in tqdm(
        range(len(sigma_schedule) - 1),
        desc="Sampling Progress",
        disable=not dist.is_initialized() or dist.get_rank() != 0,
    ):
        sigma = sigma_schedule[i]

        # print(z.shape)
        pred = v_pred_fn(z.to(dtype), sigma)
        if solver == "flow":
            z, pred_original, log_prob = flow_grpo_step(
                model_output=pred.float(),
                latents=z.float(),
                eta=eta if not determistic else 0,
                sigmas=sigma_schedule,
                index=i,
                prev_sample=None,
            )
        elif solver == "dance":
            z, pred_original, log_prob = dance_grpo_step(
                pred.float(), z.float(), eta if not determistic else 0, sigmas=sigma_schedule, index=i, prev_sample=None
            )
        elif solver == "ddim":
            z, pred_original, log_prob = ddim_step(
                pred.float(), z.float(), eta if not determistic else 0, sigmas=sigma_schedule, index=i, prev_sample=None
            )
        elif "dpm" in solver:
            assert determistic
            z, pred_original, log_prob = dpm_step(
                order,
                model_output=pred.float(),
                sample=z.float(),
                step_index=i,
                timesteps=sigma_schedule[:-1],
                sigmas=sigma_schedule,
                dpm_state=dpm_state,
            )
        else:
            assert False
        z = z.to(dtype)
        all_latents.append(z)
        all_log_probs.append(log_prob)

    latents = z.to(dtype)
    # all_latents = torch.stack(all_latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
    # all_log_probs = torch.stack(all_log_probs, dim=1)  # (batch_size, num_steps, 1)
    return latents, all_latents, all_log_probs

# def run_sampling_stepwise_cache(
#     v_pred_fn,
#     z,
#     sigma_schedule,
#     solver="flow",
#     determistic=False,
#     eta=0.7,
#     acceleration_policy=None,
#     actions=None,
# ):
#     """
#     z: latents
#     sigma: predicted noise
#     cache_actions:
#         - None: same as full compute
#         - Tensor: 0-full compute; 1-cache; decide for every single timestep
#     """
#     assert solver in ["flow", "dance", "ddim", "dpm1", "dpm2"]
#     dtype = z.dtype
#     device = z.device
#     all_latents = [z]
#     all_log_probs = []

#     num_steps = len(sigma_schedule) - 1
#     batch_size = z.shape[0]

#     # store cache
#     cached_pred = None
#     # timesteps to last full compute (in positive integer) or cache (in negative integer), using abs value
#     dis = 0
#     distance = torch.full((batch_size,), dis, device=device, dtype=dtype)
#     num_full_steps = torch.full((batch_size, 1), 0, device=device, dtype=dtype)

#     if "dpm" in solver:
#         order = int(solver[-1])
#         dpm_state = DPMState(order=order)

#     # ====================== Predict actions one at a timestep ======================
#     # loop over the timesteps
#     for i in tqdm(
#         range(num_steps),
#         desc="Sampling Progress (cached)",
#         disable=not dist.is_initialized() or dist.get_rank() != 0,
#     ):
#         sigma = sigma_schedule[i]

#         # sample actions
#         # log_probs = logΠ_θ(a_t | s_t); Π_θ: policy network, a_t: full compute or cache, s_t: timestep, latents, ...
#         current_timestep = torch.full((batch_size,), i, device=device, dtype=dtype)

#         # z: [9, 16, 64, 64]-[batch_size, latent_channel, H, W]
#         # returned action is an integer denoting the action idx
#         # 0: full compute; 1: cache; 2: skip; 3: dynamic resolution ...
#         # The training loop wraps sampling in no_grad for the diffusion model; re-enable grads
#         # here so the policy receives gradients from the log-prob term.
#         with torch.enable_grad():
#             action, policy_log_probs = acceleration_policy.sample_action(actions, current_timestep, z, distance)

#         # action: [B, 1]
#         # proceed action in batch manner using mask
#         full_mask  = (action == actions['full'])
#         cache_mask = (action == actions['cache'])

#         pred = torch.empty_like(z, dtype=dtype, device=z.device)

#         # full compute
#         if full_mask.any():
#             z_full = z[full_mask]                             # [B_full, C, H, W]
#             # print(z_full.shape)
#             pred_full = v_pred_fn(z_full.to(dtype), sigma, full_mask, current_timestep)    # [B_full, C, H, W]

#             pred[full_mask] = pred_full.to(pred.dtype)

#             # 初始化 cached_pred（第一次时）
#             if cached_pred is None:
#                 cached_pred = torch.zeros_like(pred, dtype=pred.dtype, device=pred.device)

#             # 只更新 full 的那部分 cache
#             cached_pred[full_mask] = pred_full.detach().to(cached_pred.dtype)

#             # distance 更新：distance > 0 则 +1，否则置为 1
#             # distance: [B]
#             distance_pos = distance.clone()
#             distance_pos[full_mask & (distance > 0)] += 1
#             distance_pos[full_mask & (distance <= 0)] = 1
#             distance = distance_pos

#             num_full_steps[full_mask] += 1

#         # cache
#         if cache_mask.any():
#             pred[cache_mask] = cached_pred[cache_mask]

#             distance_neg = distance.clone()
#             distance_neg[cache_mask & (distance < 0)] -= 1
#             distance_neg[cache_mask & (distance >= 0)] = -1
#             distance = distance_neg


#         # TODO: add more actions

#         # Note: latent z is updated below as the solver output
#         if solver == "flow":
#             z, pred_original, log_prob = flow_grpo_step(
#                 model_output=pred.float(),
#                 latents=z.float(),
#                 eta=eta if not determistic else 0,
#                 sigmas=sigma_schedule,
#                 index=i,
#                 prev_sample=None,
#             )
#         elif solver == "dance":
#             z, pred_original, log_prob = dance_grpo_step(
#                 pred.float(), z.float(), eta if not determistic else 0, sigmas=sigma_schedule, index=i, prev_sample=None
#             )
#         elif solver == "ddim":
#             z, pred_original, log_prob = ddim_step(
#                 pred.float(), z.float(), eta if not determistic else 0, sigmas=sigma_schedule, index=i, prev_sample=None
#             )
#         elif "dpm" in solver:
#             assert determistic
#             z, pred_original, log_prob = dpm_step(
#                 order,
#                 model_output=pred.float(),
#                 sample=z.float(),
#                 step_index=i,
#                 timesteps=sigma_schedule[:-1],
#                 sigmas=sigma_schedule,
#                 dpm_state=dpm_state,
#             )
#         else:
#             assert False
#         z = z.to(dtype)
#         all_latents.append(z)
#         all_log_probs.append(log_prob)

#     latents = z.to(dtype)
#     # all_latents = torch.stack(all_latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
#     # all_log_probs = torch.stack(all_log_probs, dim=1)  # (batch_size, num_steps, 1)
#     return latents, all_latents, all_log_probs, policy_log_probs, num_full_steps

# def run_sampling_layerwise_cache(
#     v_pred_fn,
#     z,
#     sigma_schedule,
#     solver="flow",
#     determistic=False,
#     eta=0.7,
#     acceleration_policy=None,
#     actions=None,
#     skipped_layers=None,
# ):
#     """
#     z: latents
#     sigma: predicted noise
#     cache_actions:
#         - None: same as full compute
#         - Tensor: 0-full compute; 1-cache; decide for every single timestep
#     """
#     assert solver in ["flow", "dance", "ddim", "dpm1", "dpm2"]
#     dtype = z.dtype
#     device = z.device
#     all_latents = [z]
#     all_log_probs = []

#     num_steps = len(sigma_schedule) - 1
#     batch_size = z.shape[0]

#     # timesteps to last full compute (in positive integer) or cache (in negative integer), using abs value
#     dis = 0
#     distance = torch.full((batch_size,), dis, device=device, dtype=dtype)
#     num_full_steps = torch.full((batch_size,), 0, device=device, dtype=dtype)
#     # cache[batch_id][timestep][layer_idx] = features
#     cache_dict = defaultdict(lambda: defaultdict(dict))

#     if "dpm" in solver:
#         order = int(solver[-1])
#         dpm_state = DPMState(order=order)

#     # ====================== Predict actions one at a timestep ======================
#     # loop over the timesteps
#     for i in tqdm(
#         range(num_steps),
#         desc="Sampling Progress (cached)",
#         disable=not dist.is_initialized() or dist.get_rank() != 0,
#     ):
#         sigma = sigma_schedule[i]
#         pred = torch.empty_like(z, dtype=dtype, device=z.device)

#         # sample actions
#         # log_probs = logΠ_θ(a_t | s_t); Π_θ: policy network, a_t: full compute or cache, s_t: timestep, latents, ...
#         current_timestep = torch.full((batch_size,), i, device=device, dtype=dtype)

#         # z: [9, 16, 64, 64]-[batch_size, latent_channel, H, W]
#         # returned action is an integer denoting the action idx
#         # 0: full compute; 1: cache; 2: skip; 3: dynamic resolution ...
#         # The training loop wraps sampling in no_grad for the diffusion model; re-enable grads
#         # here so the policy receives gradients from the log-prob term.
#         with torch.enable_grad():
#             action, policy_log_probs = acceleration_policy.sample_action(actions, current_timestep, z, distance)

#         # action: [B, 1]
#         # proceed action in batch manner using mask
#         full_mask  = (action == actions['full'])
#         idx_full = torch.nonzero(full_mask, as_tuple=True)[0]
#         cache_mask = (action == actions['cache'])
#         idx_cache = torch.nonzero(cache_mask, as_tuple=True)[0]


#         # full compute
#         if full_mask.any():
#             z_full = z[full_mask]                             # [B_full, C, H, W]
#             pred_full = v_pred_fn(z_full.to(dtype), sigma, full_mask, i, skipped_layers, use_cache=True, cache_dict=cache_dict, sample_idx=idx_full, action=actions['full'])    # [B_full, C, H, W]
#             pred[full_mask] = pred_full.to(pred.dtype)

#             # distance 更新：distance > 0 则 +1，否则置为 1
#             # distance: [B]
#             distance_pos = distance.clone()
#             distance_pos[full_mask & (distance > 0)] += 1
#             distance_pos[full_mask & (distance <= 0)] = 1
#             distance = distance_pos

#             num_full_steps[full_mask] += 1

#         # cache
#         if cache_mask.any():
#             z_cache = z[cache_mask]
#             pred_cache = v_pred_fn(z_cache.to(dtype), sigma, cache_mask, i, skipped_layers, use_cache=True, cache_dict=cache_dict, sample_idx=idx_cache, action=actions['cache'])
#             pred[cache_mask] = pred_cache.to(pred.dtype)

#             distance_neg = distance.clone()
#             distance_neg[cache_mask & (distance < 0)] -= 1
#             distance_neg[cache_mask & (distance >= 0)] = -1
#             distance = distance_neg

#         # TODO: add more actions

#         # Note: latent z is updated below as the solver output
#         if solver == "flow":
#             z, pred_original, log_prob = flow_grpo_step(
#                 model_output=pred.float(),
#                 latents=z.float(),
#                 eta=eta if not determistic else 0,
#                 sigmas=sigma_schedule,
#                 index=i,
#                 prev_sample=None,
#             )
#         elif solver == "dance":
#             z, pred_original, log_prob = dance_grpo_step(
#                 pred.float(), z.float(), eta if not determistic else 0, sigmas=sigma_schedule, index=i, prev_sample=None
#             )
#         elif solver == "ddim":
#             z, pred_original, log_prob = ddim_step(
#                 pred.float(), z.float(), eta if not determistic else 0, sigmas=sigma_schedule, index=i, prev_sample=None
#             )
#         elif "dpm" in solver:
#             assert determistic
#             z, pred_original, log_prob = dpm_step(
#                 order,
#                 model_output=pred.float(),
#                 sample=z.float(),
#                 step_index=i,
#                 timesteps=sigma_schedule[:-1],
#                 sigmas=sigma_schedule,
#                 dpm_state=dpm_state,
#             )
#         else:
#             assert False
#         z = z.to(dtype)
#         all_latents.append(z)
#         all_log_probs.append(log_prob)

#     latents = z.to(dtype)
#     # all_latents = torch.stack(all_latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
#     # all_log_probs = torch.stack(all_log_probs, dim=1)  # (batch_size, num_steps, 1)
#     return latents, all_latents, all_log_probs, policy_log_probs, num_full_steps

# ============================================= As of 1.13 using this ================================================
def build_action_array(interval: int, length: int = 40):
    assert interval >= 1, "interval must be >= 1"

    arr = [1] * length

    # 每隔 interval 放一个 0
    for i in range(0, length, interval):
        arr[i] = 0

    # 最后两个强制为 0
    arr[-2:] = [0, 0]

    return arr


def run_sampling_cache(
    v_pred_fn,
    z,
    sigma_schedule,
    solver="flow",
    determistic=False,
    eta=0.7,
    acceleration_policy=None,
    actions=None,
    skipped_layers=None,
    sft=None,
    max_order=2,
    interval=4,
):
    """
    z: latents
    sigma: predicted noise
    cache_actions:
        - None: same as full compute
        - Tensor: 0-full compute; 1-cache; decide for every single timestep
    """
    assert solver in ["flow", "dance", "ddim", "dpm1", "dpm2"]
    dtype = z.dtype
    device = z.device
    all_latents = [z]
    all_log_probs = []

    num_steps = len(sigma_schedule) - 1
    batch_size = z.shape[0]

    num_full_steps = 0
    # cache[timestep][layer_idx] = features
    cache_dict = defaultdict(lambda: defaultdict(dict))
    full_compute_steps = []
    # 记录每个 cache 步所采用的 ts_order
    cache_steps_order = {}
    # order_pred 模式下，action 由 build_action_array(interval) 决定
    # interval 从外部参数传入，length 使用实际采样步数
    ts_actions = build_action_array(interval=interval, length=num_steps + 1)
    # 获取 policy 模式
    policy_mode = acceleration_policy.mode if acceleration_policy is not None else 'order_pred'
    sft_loss = 0
    sft_action_loss = 0
    sft_order_loss = 0

    if "dpm" in solver:
        order = int(solver[-1])
        dpm_state = DPMState(order=order)

    activated_steps = [0, 0]

    status = {}
    status['distance'] = 0
    status['interval'] = 0

    # 累积所有步的 policy log probs
    all_policy_log_probs = []

    # ====================== Predict actions one at a timestep ======================
    # loop over the timesteps
    for i in tqdm(
        range(num_steps),
        desc="Sampling Progress (cached)",
        disable=not dist.is_initialized() or dist.get_rank() != 0,
    ):
        sigma = sigma_schedule[i]
        # print(sigma)
        # print(f'sigma shape: {sigma.shape}')
        pred = torch.empty_like(z, dtype=dtype, device=z.device)

        # sample actions
        # log_probs = logΠ_θ(a_t | s_t); Π_θ: policy network, a_t: full compute or cache, s_t: timestep, latents, ...
        # current_timestep = torch.full((batch_size,), i, device=device, dtype=dtype)
        current_timestep = torch.full((batch_size,), sigma * 1000, device=device, dtype=dtype)

        distance = torch.full((batch_size,), status['distance'], device=device, dtype=dtype)
        interval = torch.full((batch_size,), status['interval'], device=device, dtype=dtype)

        # z: [9, 16, 64, 64]-[batch_size, latent_channel, H, W]
        # The training loop wraps sampling in no_grad for the diffusion model; re-enable grads
        # here so the policy receives gradients from the log-prob term.
        with torch.enable_grad():
            if policy_mode == 'full_pred':
                # ==================== full_pred 模式：同时预测 action 和 order ====================
                if not sft:
                    action, ts_order, policy_log_probs = acceleration_policy.sample_action(
                        actions, current_timestep, i, z, distance, interval, batch_wise=True, sft=sft
                    )
                    all_policy_log_probs.append(policy_log_probs)
                    current_action = action[0].item()
                else:
                    # SFT 模式：监督 action 和 order
                    logit_action, logit_order = acceleration_policy(current_timestep, z, distance, interval, sft=sft)
                    logit_action = logit_action.mean(dim=0, keepdim=True)  # [1,2]
                    logit_order = logit_order.mean(dim=0, keepdim=True)    # [1,K]
                    teacher_action = ts_actions[i]
                    target_action = torch.tensor([teacher_action], device=z.device, dtype=torch.long)
                    loss_action = F.cross_entropy(logit_action, target_action)
                    sft_action_loss += loss_action

                    # 全计算步锁死 order=max_order，不需要监督；Cache 步需要学习 order 决策
                    if teacher_action == actions['full']:
                        loss_step = loss_action
                        target_order = torch.tensor([max_order], device=z.device, dtype=torch.long)
                    else:
                        target_order = torch.tensor([max_order], device=z.device, dtype=torch.long)
                        loss_order = F.cross_entropy(logit_order, target_order)
                        sft_order_loss += loss_order
                        loss_step = loss_action + loss_order

                    sft_loss += loss_step
                    action = torch.full((batch_size,), teacher_action, device=device, dtype=torch.long)
                    current_action = teacher_action
                    ts_order = target_order
            elif policy_mode == 'order_pred':
                # ==================== order_pred 模式：只预测 order，action 由 ts_actions 决定 ====================
                current_action = ts_actions[i]  # 从预定义的 action 序列中获取
                action = torch.full((batch_size,), current_action, device=device, dtype=torch.long)

                if not sft:
                    ts_order, policy_log_probs = acceleration_policy.sample_order(
                        current_action, actions, current_timestep, i, z, distance, interval, batch_wise=True, sft=sft
                    )
                    all_policy_log_probs.append(policy_log_probs)
                else:
                    # SFT 模式：只监督 order，action 由 ts_actions 决定
                    logit_order = acceleration_policy(current_timestep, z, distance, interval, sft=sft)
                    logit_order = logit_order.mean(dim=0, keepdim=True)   # [1,K]

                    # 全计算步锁死 order=max_order，不需要监督；Cache 步需要学习 order 决策
                    if current_action == actions['full']:
                        target_order = torch.tensor([max_order], device=z.device, dtype=torch.long)
                        # full compute 时不计算 order loss
                    else:
                        target_order = torch.tensor([max_order], device=z.device, dtype=torch.long)
                        loss_order = F.cross_entropy(logit_order, target_order)
                        sft_order_loss += loss_order
                        sft_loss += loss_order

                    ts_order = target_order
            else:
                # ==================== action_pred 模式：只预测 action，order 固定为 max_order ====================
                if not sft:
                    action, ts_order, policy_log_probs = acceleration_policy.sample_action_only(
                        actions, current_timestep, i, z, distance, interval, batch_wise=True, sft=sft
                    )
                    all_policy_log_probs.append(policy_log_probs)
                    current_action = action[0].item()
                else:
                    # SFT 模式：只监督 action，order 固定为 max_order
                    logit_action = acceleration_policy(current_timestep, z, distance, interval, sft=sft)
                    logit_action = logit_action.mean(dim=0, keepdim=True)  # [1,2]
                    teacher_action = ts_actions[i]
                    target_action = torch.tensor([teacher_action], device=z.device, dtype=torch.long)
                    loss_action = F.cross_entropy(logit_action, target_action)
                    sft_action_loss += loss_action
                    sft_loss += loss_action

                    action = torch.full((batch_size,), teacher_action, device=device, dtype=torch.long)
                    current_action = teacher_action
                    ts_order = torch.tensor([max_order], device=z.device, dtype=torch.long)


        if current_action == actions['full']:
            full_compute_steps.append(i)
            num_full_steps += 1

            status['max_order'] = max_order # 全计算步保证得到所有高阶近似

            activated_steps[0] = activated_steps[-1]
            activated_steps[-1] = i
            status['interval'] = activated_steps[-1] - activated_steps[0]
            # Only keep gradients for policy log-prob; diffusion forward should not build a graph.
            with torch.no_grad():
                pred = v_pred_fn(z.to(dtype), sigma, i, skipped_layers, cache_dict=cache_dict, status=status, action=actions['full'])

            status['last_full_compute'] = i
            status['distance'] = 1

            # 清理旧缓存，节省显存
            cleanup_cache(cache_dict, i, status['interval'])

        # cache
        else:
            status['max_order'] = int(ts_order[0].item())
            # 记录当前 cache 步的 ts_order（仅在非 SFT 模式下记录，SFT 时是教师目标不是模型预测）
            if not sft:
                cache_steps_order[i] = int(ts_order[0].item())

            # Only keep gradients for policy log-prob; diffusion forward should not build a graph.
            with torch.no_grad():
                pred = v_pred_fn(z.to(dtype), sigma, i, skipped_layers, cache_dict=cache_dict, status=status, action=actions['cache'])

            # cache就加一步与上一全计算步的距离
            status['distance'] += 1

        # TODO: add more actions

        # Note: latent z is updated below as the solver output
        if solver == "flow":
            z, pred_original, log_prob = flow_grpo_step(
                model_output=pred.float(),
                latents=z.float(),
                eta=eta if not determistic else 0,
                sigmas=sigma_schedule,
                index=i,
                prev_sample=None,
            )
        elif solver == "dance":
            z, pred_original, log_prob = dance_grpo_step(
                pred.float(), z.float(), eta if not determistic else 0, sigmas=sigma_schedule, index=i, prev_sample=None
            )
        elif solver == "ddim":
            z, pred_original, log_prob = ddim_step(
                pred.float(), z.float(), eta if not determistic else 0, sigmas=sigma_schedule, index=i, prev_sample=None
            )
        elif "dpm" in solver:
            assert determistic
            z, pred_original, log_prob = dpm_step(
                order,
                model_output=pred.float(),
                sample=z.float(),
                step_index=i,
                timesteps=sigma_schedule[:-1],
                sigmas=sigma_schedule,
                dpm_state=dpm_state,
            )
        else:
            assert False
        z = z.to(dtype)
        all_latents.append(z)
        all_log_probs.append(log_prob)

    sft_loss = sft_loss / num_steps
    # 确保返回的是 tensor，即使在 order_pred 模式下 action_loss 为 0
    if isinstance(sft_action_loss, (int, float)):
        sft_action_loss = torch.tensor(sft_action_loss, device=device)
    else:
        sft_action_loss = sft_action_loss / num_steps
    if isinstance(sft_order_loss, (int, float)):
        sft_order_loss = torch.tensor(sft_order_loss, device=device)
    else:
        sft_order_loss = sft_order_loss / num_steps
    latents = z.to(dtype)
    # all_latents = torch.stack(all_latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
    # all_log_probs = torch.stack(all_log_probs, dim=1)  # (batch_size, num_steps, 1)
    if not sft:
        # 将所有步的 policy log_probs 堆叠成 [num_steps, batch_size] 的张量
        acceleration_policy_log_probs = torch.stack(all_policy_log_probs, dim=0)
        return latents, all_latents, all_log_probs, acceleration_policy_log_probs, num_full_steps, full_compute_steps, cache_steps_order
    else:
        sft_losses = {
            'total': sft_loss,
            'action': sft_action_loss,
            'order': sft_order_loss,
        }
        return latents, all_latents, all_log_probs, sft_losses

# =========================== 用来测量在给定order和interval时模型表现的 ===========================
def run_sampling_cache_test(
    v_pred_fn,
    z,
    sigma_schedule,
    solver="flow",
    determistic=False,
    eta=0.7,
    actions=None,
    skipped_layers=None,
    max_order=2,
    interval=5,
):
    """
    z: latents
    sigma: predicted noise
    cache_actions:
        - None: same as full compute
        - Tensor: 0-full compute; 1-cache; decide for every single timestep
    """
    assert solver in ["flow", "dance", "ddim", "dpm1", "dpm2"]
    dtype = z.dtype
    device = z.device
    all_latents = [z]
    all_log_probs = []

    num_steps = len(sigma_schedule) - 1
    batch_size = z.shape[0]

    cache_dict = defaultdict(lambda: defaultdict(dict))
    teacher_actions = [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0]

    if "dpm" in solver:
        order = int(solver[-1])
        dpm_state = DPMState(order=order)

    status = {}
    status['interval'] = interval
    status['max_order'] = max_order
    status['distance'] = 0
    status['last_full_compute'] = 0

    for i in tqdm(
        range(num_steps),
        desc="Sampling Progress (cached)",
        disable=not dist.is_initialized() or dist.get_rank() != 0,
    ):
        sigma = sigma_schedule[i]
        pred = torch.empty_like(z, dtype=dtype, device=z.device)

        teacher_action = teacher_actions[i]
        action = torch.full((batch_size,), teacher_action, device=device, dtype=dtype)

        if action[0] == actions['full']:
            # 先计算动态 interval（用于 derivative_approximation）
            if i > 0:
                status['interval'] = i - status['last_full_compute']

            with torch.no_grad():
                pred = v_pred_fn(z.to(dtype), sigma, i, skipped_layers, cache_dict=cache_dict, status=status, action=actions['full'])

            # 更新状态
            status['last_full_compute'] = i
            status['distance'] = 1

            # 清理旧缓存，节省显存
            cleanup_cache(cache_dict, i, status['interval'])

            # if dist.get_rank() == 0:
            #     print('********************** Full compute **********************')
            #     print(f"step={i}, interval={status['interval']}, cache_keys={list(cache_dict.keys())}")
            #     print(f"taylor_orders: {cache_dict[i][0]['attn'].keys()}")
        # cache
        else:
            with torch.no_grad():
                pred = v_pred_fn(z.to(dtype), sigma, i, skipped_layers, cache_dict=cache_dict, status=status, action=actions['cache'])
                status['distance'] += 1
                # if dist.get_rank() == 0:

        # Note: latent z is updated below as the solver output
        if solver == "flow":
            z, pred_original, log_prob = flow_grpo_step(
                model_output=pred.float(),
                latents=z.float(),
                eta=eta if not determistic else 0,
                sigmas=sigma_schedule,
                index=i,
                prev_sample=None,
            )
        elif solver == "dance":
            z, pred_original, log_prob = dance_grpo_step(
                pred.float(), z.float(), eta if not determistic else 0, sigmas=sigma_schedule, index=i, prev_sample=None
            )
        elif solver == "ddim":
            z, pred_original, log_prob = ddim_step(
                pred.float(), z.float(), eta if not determistic else 0, sigmas=sigma_schedule, index=i, prev_sample=None
            )
        elif "dpm" in solver:
            assert determistic
            z, pred_original, log_prob = dpm_step(
                order,
                model_output=pred.float(),
                sample=z.float(),
                step_index=i,
                timesteps=sigma_schedule[:-1],
                sigmas=sigma_schedule,
                dpm_state=dpm_state,
            )
        else:
            assert False
        z = z.to(dtype)
        all_latents.append(z)
        all_log_probs.append(log_prob)

    latents = z.to(dtype)
    # all_latents = torch.stack(all_latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
    # all_log_probs = torch.stack(all_log_probs, dim=1)  # (batch_size, num_steps, 1)
    return latents, all_latents, all_log_probs

def flow_grpo_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: torch.Tensor,
    generator: Optional[torch.Generator] = None,
):
    device = model_output.device
    sigma = sigmas[index].to(device)
    sigma_prev = sigmas[index + 1].to(device)
    sigma_max = sigmas[1].item()
    dt = sigma_prev - sigma  # neg dt

    pred_original_sample = latents - sigma * model_output

    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * eta

    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )

    prev_sample_mean = (
        latents * (1 + std_dev_t**2 / (2 * sigma) * dt)
        + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
    )

    if prev_sample is None:
        variance_noise = randn_tensor(model_output.shape, generator=generator, device=device, dtype=model_output.dtype)
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise

    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2))
        - torch.log(std_dev_t * torch.sqrt(-1 * dt))
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )

    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample, pred_original_sample, log_prob


def dance_grpo_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: torch.Tensor,
):
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma  # neg dt
    prev_sample_mean = latents + dsigma * model_output

    pred_original_sample = latents - sigma * model_output

    delta_t = sigma - sigmas[index + 1]  # pos -dt
    std_dev_t = eta * math.sqrt(delta_t)

    score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma**2
    log_term = -0.5 * eta**2 * score_estimate
    prev_sample_mean = prev_sample_mean + log_term * dsigma

    if prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

    # log prob of prev_sample given prev_sample_mean and std_dev_t
    log_prob = -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (
        2 * (std_dev_t**2)
    )
    -math.log(std_dev_t) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))

    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    return prev_sample, pred_original_sample, log_prob


def ddim_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: torch.Tensor,
):
    model_output = convert_model_output(model_output, latents, sigmas, step_index=index)
    prev_sample, prev_sample_mean, std_dev_t, dt_sqrt = ddim_update(
        model_output,
        sigmas.to(torch.float64),
        index,
        latents,
        eta=eta,
    )

    # Compute log_prob
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * dt_sqrt) ** 2))
        - torch.log(std_dev_t * dt_sqrt)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )

    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    return prev_sample, model_output, log_prob


@dataclass
class DPMState:
    order: int
    model_outputs: List[torch.Tensor] = None
    lower_order_nums = 0

    def __post_init__(self):
        self.model_outputs = [None] * self.order

    def update(self, model_output: torch.Tensor):
        for i in range(self.order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

    def update_lower_order(self):
        if self.lower_order_nums < self.order:
            self.lower_order_nums += 1


def dpm_step(
    order,
    model_output: torch.Tensor,
    sample: torch.Tensor,
    step_index: int,
    timesteps: list,
    sigmas: torch.Tensor,
    dpm_state: DPMState = None,
) -> torch.Tensor:

    # Improve numerical stability for small number of steps
    lower_order_final = step_index == len(timesteps) - 1
    lower_order_second = (step_index == len(timesteps) - 2) and len(timesteps) < 15

    model_output = convert_model_output(model_output, sample, sigmas, step_index=step_index)

    assert dpm_state is not None
    dpm_state.update(model_output)

    # Upcast to avoid precision issues when computing prev_sample
    sample = sample.to(torch.float32)

    if order == 1 or dpm_state.lower_order_nums < 1 or lower_order_final:
        if step_index == 0 or lower_order_final:
            prev_sample, _, _, _ = ddim_update(
                model_output,
                sigmas.to(torch.float64),
                step_index,
                sample,
                eta=0.0,
            )
        else:
            prev_sample = dpm_solver_first_order_update(
                model_output,
                sigmas.to(torch.float64),
                step_index,
                sample,
            )
    elif order == 2 or dpm_state.lower_order_nums < 2 or lower_order_second:
        prev_sample = multistep_dpm_solver_second_order_update(
            dpm_state.model_outputs,
            sigmas.to(torch.float64),
            step_index,
            sample,
        )
    else:
        assert False

    dpm_state.update_lower_order()

    # Cast sample back to expected dtype
    prev_sample = prev_sample.to(model_output.dtype)

    return prev_sample, model_output, None


def convert_model_output(
    model_output,
    sample,
    sigmas,
    step_index,
) -> torch.Tensor:
    sigma_t = sigmas[step_index]
    x0_pred = sample - sigma_t * model_output

    return x0_pred


def ddim_update(
    model_output: torch.Tensor,
    sigmas,
    step_index,
    sample: torch.Tensor = None,
    noise: Optional[torch.Tensor] = None,
    eta: float = 1.0,
) -> torch.Tensor:

    t, s = sigmas[step_index + 1], sigmas[step_index]

    std_dev_t = eta * t
    dt_sqrt = torch.sqrt(1.0 - t**2 * (1 - s) ** 2 / (s**2 * (1 - t) ** 2))
    rho_t = std_dev_t * dt_sqrt
    noise_pred = (sample - (1 - s) * model_output) / s
    if noise is None:
        noise = torch.randn_like(model_output)
    prev_mean = (1 - t) * model_output + torch.sqrt(t**2 - rho_t**2) * noise_pred
    x_t = prev_mean + rho_t * noise

    return x_t, prev_mean, std_dev_t, dt_sqrt


def dpm_solver_first_order_update(
    model_output: torch.Tensor,
    sigmas,
    step_index,
    sample: torch.Tensor = None,
) -> torch.Tensor:

    sigma_t, sigma_s = sigmas[step_index + 1], sigmas[step_index]
    alpha_t, sigma_t = _sigma_to_alpha_sigma_t(sigma_t)
    alpha_s, sigma_s = _sigma_to_alpha_sigma_t(sigma_s)
    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s = torch.log(alpha_s) - torch.log(sigma_s)

    h = lambda_t - lambda_s
    x_t = (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * model_output

    return x_t


def multistep_dpm_solver_second_order_update(
    model_output_list: List[torch.Tensor],
    sigmas,
    step_index,
    sample: torch.Tensor = None,
) -> torch.Tensor:

    sigma_t, sigma_s0, sigma_s1 = (
        sigmas[step_index + 1],
        sigmas[step_index],
        sigmas[step_index - 1],
    )

    alpha_t, sigma_t = _sigma_to_alpha_sigma_t(sigma_t)
    alpha_s0, sigma_s0 = _sigma_to_alpha_sigma_t(sigma_s0)
    alpha_s1, sigma_s1 = _sigma_to_alpha_sigma_t(sigma_s1)

    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
    lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)

    m0, m1 = model_output_list[-1], model_output_list[-2]

    h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
    r0 = h_0 / h
    D0, D1 = m0, (1.0 / r0) * (m0 - m1)

    x_t = (
        (sigma_t / sigma_s0) * sample
        - (alpha_t * (torch.exp(-h) - 1.0)) * D0
        - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
    )

    return x_t


def _sigma_to_alpha_sigma_t(sigma):
    alpha_t = 1 - sigma
    sigma_t = sigma
    return alpha_t, sigma_t
