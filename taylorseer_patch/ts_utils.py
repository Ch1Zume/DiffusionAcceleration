from typing import Dict
import torch
import math

def forecast(f: torch.Tensor, f_last: torch.Tensor, N: int, k: int) -> torch.Tensor:
    delta_f = (f - f_last) / N
    f_k = f + (k * delta_f) / N
    return f_k

# def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
#     """
#     Compute derivative approximation.
#     :param cache_dic: Cache dictionary.
#     :param current: Current step information.
#     """
#     difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]
#     # difference_distance = current['activated_times'][-1] - current['activated_times'][-2]

#     updated_taylor_factors = {}
#     updated_taylor_factors[0] = feature # zero order feature

#     for i in range(cache_dic['max_order']):
#         if (cache_dic['cache'][-1][current['layer']][current['module']].get(i, None) is not None) and (current['step'] < (current['num_steps'] - cache_dic['first_enhance'] + 1)):
#             updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - cache_dic['cache'][-1][current['layer']][current['module']][i]) / difference_distance
#         else:
#             break

#     cache_dic['cache'][-1][current['layer']][current['module']] = updated_taylor_factors

# def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor:
#     """
#     Compute Taylor expansion error.
#     :param cache_dic: Cache dictionary.
#     :param current: Current step information.
#     """
#     x = current['step'] - current['activated_steps'][-1]
#     # x = current['t'] - current['activated_times'][-1]
#     output = 0

#     for i in range(len(cache_dic['cache'][-1][current['layer']][current['module']])):
#         output += (1 / math.factorial(i)) * cache_dic['cache'][-1][current['layer']][current['module']][i] * (x ** i)

#     return output

# def taylor_cache_init(cache_dic: Dict, current: Dict):
#     """
#     Initialize Taylor cache and expand storage for different-order derivatives.
#     :param cache_dic: Cache dictionary.
#     :param current: Current step information.
#     """
#     if current['step'] == (current['num_steps'] - 1):
#         cache_dic['cache'][-1][current['layer']][current['module']] = {}

def derivative_approximation(cache_dict: Dict, feature: torch.Tensor, module: str, max_order: int, layer_idx: int, interval: int, step: int):
    """
    在全计算步中计算并存储 Taylor 系数（各阶导数近似）。
    只在此函数中创建 cache_dict 的键值对。

    :param cache_dict: 缓存字典
    :param feature: 当前模块的输出特征
    :param module: 模块名称 ('attn', 'context_attn', 'attn2', 'ff')
    :param max_order: 最大 Taylor 阶数
    :param layer_idx: 层索引
    :param interval: 全计算步之间的间隔
    :param step: 当前步数
    """
    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature  # 零阶项（特征本身）
    prev_step = step - interval

    for i in range(max_order):
        # 在满足以下条件时计算高阶项：
        # 1. 有前一步的高阶项（需要最初的几个全计算步来积累）
        # 2. 不是最后几个step（为了生成质量最后几个step强制全计算）
        prev_cache = cache_dict.get(prev_step, {}).get(layer_idx, {}).get(module, {})
        if (prev_cache.get(i, None) is not None) and (step <= 37):
            updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - prev_cache[i]) / interval
        else:
            break

    # 只在全计算步创建键值对
    if step not in cache_dict:
        cache_dict[step] = {}
    if layer_idx not in cache_dict[step]:
        cache_dict[step][layer_idx] = {}
    cache_dict[step][layer_idx][module] = updated_taylor_factors


def taylor_formula(cache_dict: Dict, distance, module, layer_idx: int, last_full_step: int) -> torch.Tensor:
    """
    使用 Taylor 展开公式计算缓存步的输出。
    使用 .get() 安全读取，不创建新的键值对。

    :param cache_dict: 缓存字典
    :param distance: 与最近全计算步的距离
    :param module: 模块名称
    :param layer_idx: 层索引
    :param last_full_step: 最近的全计算步
    :return: Taylor 展开计算的输出
    """
    output = 0
    # 使用 .get() 安全读取，不会创建新的键值对
    taylor_cache = cache_dict.get(last_full_step, {}).get(layer_idx, {}).get(module, {})

    for i in range(len(taylor_cache)):
        output += (1 / math.factorial(i)) * taylor_cache[i] * (distance ** i)

    return output

def cleanup_cache(cache_dict: Dict, current_step: int, interval: int):
    """
    清理不再需要的缓存，节省显存。

    在全计算步完成后调用。此时：
    - current_step 的缓存刚存储，后续 cache 步的 taylor_formula 需要
    - current_step - interval 的缓存刚被 derivative_approximation 读取，之后不再需要

    :param cache_dict: 缓存字典
    :param current_step: 当前全计算步
    :param interval: 全计算步之间的间隔
    """
    if interval <= 0:
        return  # 第一步时 interval 可能为 0，无需清理

    # 删除 current_step - interval 及更早的缓存
    # 只保留 current_step 的缓存给后续 cache 步使用
    threshold = current_step - interval + 1  # 删除所有 step < threshold 的缓存

    # 找出需要删除的 keys（只处理非负整数 key）
    keys_to_delete = [k for k in list(cache_dict.keys()) if isinstance(k, int) and k >= 0 and k < threshold]

    for k in keys_to_delete:
        del cache_dict[k]