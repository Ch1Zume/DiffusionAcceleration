from collections import defaultdict
import os
import sys
import math
import datetime
from concurrent import futures
import time
import json
from absl import app, flags
import logging
from diffusers import StableDiffusion3Pipeline
import numpy as np
from regex import R
from torch.cuda import device_count
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob, pipeline_with_logprob_layerwise_cache, pipeline_with_logprob_layerwise_cache_batch
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper
from ml_collections import config_flags
from torch.cuda.amp import GradScaler, autocast as torch_autocast
from modules.policy import Policy, save_policy, load_policy
from taylorseer.Custom_DiT_linear import SD3Transformer2DModel_Taylor

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def setup_distributed(rank, lock_rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(lock_rank)


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

class TextPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}.txt")
        with open(self.file_path, "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}_metadata.jsonl")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        assert (
            self.total_samples % self.k == 0
        ), f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]

            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def gather_tensor_to_all(tensor, world_size):
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0).cpu()


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length)
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds

def save_image(img_tensor, filename):
    """
    img_tensor: [C,H,W], 值域 0~1 或 0~255 均可
    filename: 保存路径，例如 "output.png"
    """
    img = img_tensor.detach().cpu().numpy()

    # 如果是 0~1，转成 0~255
    if img.max() <= 1.0:
        img = img * 255

    img = img.astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC

    Image.fromarray(img).save(filename)

def eval_fn(
    pipeline,
    test_dataloader,
    text_encoders,
    tokenizers,
    config,
    device,
    rank,
    world_size,
    global_step,
    reward_fn,
    executor,
    mixed_precision_dtype,
):
    pipeline.transformer.eval()

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=device
    )

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    all_rewards = defaultdict(list)

    test_sampler = (
        DistributedSampler(test_dataloader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )
    eval_loader = DataLoader(
        test_dataloader.dataset,
        batch_size=config.sample.test_batch_size,  # This is per-GPU batch size
        sampler=test_sampler,
        collate_fn=test_dataloader.collate_fn,
        num_workers=test_dataloader.num_workers,
    )

    for test_batch in tqdm(
        eval_loader,
        desc="Eval: ",
        disable=not is_main_process(rank),
        position=0,
    ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, text_encoders, tokenizers, max_sequence_length=128, device=device
        )
        current_batch_size = len(prompt_embeds)
        if current_batch_size < len(sample_neg_prompt_embeds):  # Handle last batch
            current_sample_neg_prompt_embeds = sample_neg_prompt_embeds[:current_batch_size]
            current_sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:current_batch_size]
        else:
            current_sample_neg_prompt_embeds = sample_neg_prompt_embeds
            current_sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds

        with torch_autocast(enabled=(config.mixed_precision in ["fp16", "bf16"]), dtype=mixed_precision_dtype):
            with torch.no_grad():
                images, _, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=current_sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=current_sample_neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=config.sample.noise_level,
                    deterministic=True,
                    solver="flow",
                    model_type="sd3",
                )
        if is_main_process(rank):
            save_image(images[0], '/work/SJTU/DiffusionAcceleration/test_outputs/eval_output.png')

        rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        time.sleep(0)
        rewards, reward_metadata = rewards_future.result()

        for key, value in rewards.items():
            rewards_tensor = torch.as_tensor(value, device=device).float()
            gathered_value = gather_tensor_to_all(rewards_tensor, world_size)
            all_rewards[key].append(gathered_value.numpy())

    if is_main_process(rank):
        final_rewards = {key: np.concatenate(value_list) for key, value_list in all_rewards.items()}

        images_to_log = images.cpu()
        prompts_to_log = prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples_to_log = min(15, len(images_to_log))
            for idx in range(num_samples_to_log):
                image = images_to_log[idx].float()
                pil = Image.fromarray((image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

            sampled_prompts_log = [prompts_to_log[i] for i in range(num_samples_to_log)]
            sampled_rewards_log = [{k: final_rewards[k][i] for k in final_rewards} for i in range(num_samples_to_log)]

            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | "
                            + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts_log, sampled_rewards_log))
                    ],
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in final_rewards.items()},
                },
                step=global_step,
            )

    if world_size > 1:
        dist.barrier()

def main(_):
    config = FLAGS.config

    # --- Distributed Setup ---
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    setup_distributed(rank, local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # --- WandB Init (only on main process) ---
    if is_main_process(rank):
        log_dir = os.path.join(config.logdir, config.run_name)
        os.makedirs(log_dir, exist_ok=True)
        wandb.init(project="flow-grpo", name=config.run_name, config=config.to_dict(), dir=log_dir)
    if is_main_process(rank):
        logger.info(f"\n{config}")

    set_seed(config.seed, rank)  # Pass rank for different seeds per process

    # --- Mixed Precision Setup ---
    mixed_precision_dtype = None
    if config.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16

    enable_amp = mixed_precision_dtype is not None
    # scaler = GradScaler(enabled=enable_amp)

    # --- Load pipeline and models ---
    pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model, torch_dtype=mixed_precision_dtype)
    pipeline = pipeline.to(device) 
    # test_prompt = 'a sideview of a red colored car.'
    # with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
    #     image = pipeline(
    #         prompt=test_prompt,
    #         negative_prompt="",
    #         height=config.resolution, 
    #         width=config.resolution, 
    #         num_inference_steps=config.sample.num_steps, 
    #         guidance_scale=config.sample.guidance_scale,
    #     ).images[0]
    # image.save(f"test_outputs/official_output_initialized.png")
    if is_main_process(rank):
        logger.info("***** Model Initialized *****")

    # with cache
    base_transformer = pipeline.transformer
    config_dict = pipeline.transformer.config
    new_transformer = SD3Transformer2DModel_Taylor.from_config(config_dict)
    new_transformer.load_state_dict(base_transformer.state_dict())
    new_transformer = PeftModel.from_pretrained(new_transformer, "jieliu/SD3.5M-FlowGRPO-GenEval").merge_and_unload()
    pipeline.transformer = new_transformer.to(device, dtype=mixed_precision_dtype).eval()
    pipeline.to(device)
    if is_main_process(rank):
        logger.info("***** Transformer with TaylorSeer Loaded *****")

    # without cache
    # pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, "jieliu/SD3.5M-FlowGRPO-GenEval")
    # pipeline.transformer = pipeline.transformer.merge_and_unload()
    # pipeline.transformer.eval()
    # pipeline = pipeline.to(device)

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not is_main_process(rank),
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32

    pipeline.vae.to(device, dtype=torch.float32)  # VAE usually fp32
    pipeline.text_encoder.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_2.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_3.to(device, dtype=text_encoder_dtype)

    # transformer = pipeline.transformer.to(device)
    # DDP is not needed when a module doesn't have any parameter that requires a gradient.
    # transformer_ddp = DDP(transformer, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    if config.base_model == 'sd3':
        num_channels_latents = pipeline.transformer.config.in_channels
    else:
        num_channels_latents = pipeline.transformer.config.in_channels // 4

    # =================================================== Pretrained model test ===================================================
    # Official way from diffuser
    # with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
    #     image = pipeline(
    #         prompt=test_prompt,
    #         negative_prompt="",
    #         height=config.resolution, 
    #         width=config.resolution, 
    #         num_inference_steps=config.sample.num_steps, 
    #         guidance_scale=config.sample.guidance_scale,
    #         ).images[0]  
    # image.save(f"test_outputs/official_output_pretrained.png")
    # sys.exit()

    # Pipeline with logprob
    # prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
    #     [test_prompt], text_encoders, tokenizers, max_sequence_length=128, device=device
    # )
    # neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
    #     [""], text_encoders, tokenizers, max_sequence_length=128, device=device
    # )
    # with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
    #     with torch.no_grad():
    #         images, _, _, _ = pipeline_with_logprob(
    #             pipeline,
    #             prompt_embeds=prompt_embeds,
    #             pooled_prompt_embeds=pooled_prompt_embeds,
    #             negative_prompt_embeds=neg_prompt_embed,
    #             negative_pooled_prompt_embeds=neg_pooled_prompt_embed,
    #             num_inference_steps=config.sample.num_steps,
    #             guidance_scale=config.sample.guidance_scale,
    #             output_type="pt",
    #             height=config.resolution,
    #             width=config.resolution,
    #             noise_level=config.sample.noise_level,
    #             deterministic=config.sample.deterministic,
    #             solver=config.sample.solver,
    #             model_type="sd3",
    #         )
    # save_image(images[0], 'test_outputs/nft_output.png')
    # logger.info("***** Pretrained Model Test Finished *****")
    # sys.exit()
    # =================================================== Pretrained model test ===================================================

    if config.sft: # initialization and SFT
        acceleration_policy = Policy(
            T = config.sample.num_steps,
            num_actions = len(config.ACTIONS),
            latent_channel = num_channels_latents,
            hidden_dim = config.hidden_dim,
            ).to(device)
    else: # load pretrained network
        acceleration_policy = load_policy('ckpt_policy/0.7_180.pt').to(device)
    if is_main_process(rank):
        logger.info("***** Pre-trained Policy Network Loaded *****")

    acceleration_policy_ddp = DDP(acceleration_policy, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    policy_trainable_parameters = list(filter(lambda p: p.requires_grad, acceleration_policy_ddp.module.parameters()))

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        policy_trainable_parameters,  # Use params from original model for optimizer
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # --- Datasets and Dataloaders ---
    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, "train")
        test_dataset = TextPromptDataset(config.dataset, "test")
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, "train")
        test_dataset = GenevalPromptDataset(config.dataset, "test")
    else:
        raise NotImplementedError("Prompt function not supported with dataset")

    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,  # This is per-GPU batch size
        k=config.sample.num_image_per_prompt,
        num_replicas=world_size,
        rank=rank,
        seed=config.seed,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_sampler=train_sampler, num_workers=0, collate_fn=train_dataset.collate_fn, pin_memory=True
    )

    test_sampler = (
        DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,  # Per-GPU
        sampler=test_sampler,  # Use distributed sampler for eval
        collate_fn=test_dataset.collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=device
    )
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    # train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    # train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    samples_per_epoch = config.sample.train_batch_size * world_size * config.sample.num_batches_per_epoch
    total_train_batch_size = config.train.batch_size * world_size * config.train.gradient_accumulation_steps

    # 异步计算奖励分数
    executor = futures.ThreadPoolExecutor(max_workers=8)  # Async reward computation

    eval_reward_fn = getattr(flow_grpo.rewards, "multi_score")(device, config.reward_fn)  # Pass device
    # eval_fn(
    #     pipeline,
    #     test_dataloader,
    #     text_encoders,
    #     tokenizers,
    #     config,
    #     device,
    #     rank,
    #     world_size,
    #     eval_reward_fn,
    #     executor,
    #     mixed_precision_dtype,
    #     )

    if is_main_process(rank):
        logger.info("***** Running RL Optimization *****")
        logger.info(f"  Num Epochs = {config.num_epochs}")
        logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
        logger.info(f"  Train batch size per device = {config.train.batch_size}")
        logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
        logger.info("")
        logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")

    first_epoch = 0
    global_step = 0

    ema = None
    # smoothing
    if config.train.ema:
        ema = EMAModuleWrapper(policy_trainable_parameters, decay=0.9, update_step_interval=1, device=device)

    train_iter = iter(train_dataloader)
    optimizer.zero_grad()

    # when to stop SFT from overfitting
    threshold = -math.log(0.7)
    sft_stop_patience = 0
    should_stop = False

    for epoch in range(first_epoch, config.num_epochs):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        pipeline.transformer.eval()
        reward_fn = getattr(flow_grpo.rewards, "multi_score")(device, config.reward_fn)  # Pass device
        # samples_data_list = []

        if is_main_process(rank):
            logger.info("***** Start sampling *****")
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not is_main_process(rank),
            position=0,
        ):
            if hasattr(train_sampler, "set_epoch") and isinstance(train_sampler, DistributedKRepeatSampler):
                train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)

            prompts, prompt_metadata = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, text_encoders, tokenizers, max_sequence_length=128, device=device
            )
            prompt_ids = tokenizers[0](
                prompts, padding="max_length", max_length=256, truncation=True, return_tensors="pt"
            ).input_ids.to(device)

            # if i == 0 and epoch % 10 == 0:
            #     eval_fn(
            #         pipeline,
            #         test_dataloader,
            #         text_encoders,
            #         tokenizers,
            #         config,
            #         device,
            #         rank,
            #         world_size,
            #         global_step,
            #         eval_reward_fn,
            #         executor,
            #         mixed_precision_dtype,
            #     )

            if not config.sft:
                with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                    with torch.no_grad():
                        if is_main_process(rank) and global_step % config.log_interval == 0:
                            start_time = time.perf_counter()

                        # inference with full compute policy
                        images, _, _, prepared_latents = pipeline_with_logprob(
                            pipeline,
                            prompt_embeds=prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            negative_prompt_embeds=sample_neg_prompt_embeds[: len(prompts)],
                            negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds[: len(prompts)],
                            num_inference_steps=config.sample.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                            output_type="pt",
                            height=config.resolution,
                            width=config.resolution,
                            noise_level=config.sample.noise_level,
                            deterministic=config.sample.deterministic,
                            solver=config.sample.solver,
                            model_type="sd3",
                        )

                        if is_main_process(rank):
                            wandb.log({
                                "sample/full_compute": wandb.Image(
                                    images[0],
                                    caption=prompts[0]
                                    )
                            },
                            step=global_step,
                            )

                        if is_main_process(rank) and global_step % config.log_interval == 0:
                            end_time_full = time.perf_counter()

                    with torch.enable_grad():
                        # inference with policy with cache
                        images_cache, _, _, acceleration_policy_log_probs, num_full_steps, full_compute_steps = pipeline_with_logprob_layerwise_cache_batch(
                            pipeline,
                            prompt_embeds=prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            negative_prompt_embeds=sample_neg_prompt_embeds[: len(prompts)],
                            negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds[: len(prompts)],
                            num_inference_steps=config.sample.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                            output_type="pt",
                            height=config.resolution,
                            width=config.resolution,
                            noise_level=config.sample.noise_level,
                            deterministic=config.sample.deterministic,
                            solver=config.sample.solver,
                            model_type="sd3",
                            acceleration_policy=acceleration_policy_ddp.module,
                            actions=config.ACTIONS,
                            latents = prepared_latents, # use same initialization
                            skipped_layers = None,
                            sft = config.sft,
                        )
                        if is_main_process(rank):
                            save_dir = "full_compute_log"
                            os.makedirs(save_dir, exist_ok=True)  

                            save_path = os.path.join(save_dir, f"{global_step}.json") 

                            with open(save_path, "w") as f:
                                json.dump(full_compute_steps, f, indent=2)

                            wandb.log({
                                "sample/cached": wandb.Image(
                                    images_cache[0],
                                    caption=prompts[0]
                                    )
                            },
                            step=global_step,
                            )

                        if is_main_process(rank) and global_step % config.log_interval == 0:
                            end_time_accel = time.perf_counter()

                            t_full = end_time_full - start_time
                            t_accel = end_time_accel - end_time_full
                            accel_ratio = t_full / t_accel
                            # Policy assigns different caching strategy to each sample within the batch
                            reduced_steps = int(config.sample.num_steps - num_full_steps)
                            # 初始化时全计算和cache的步数比大概为1:1
                            
                            logger.info(f"[Timer] Sampling with full compute: {t_full:.6f} s")
                            logger.info(f"[Timer] Sampling with acceleration: {t_accel:.6f} s")
                            logger.info(f"[Timer] Acceleration Ratio: {accel_ratio:.6f}")
                            logger.info(f"[Averaged] Full compute steps: {int(num_full_steps)} | Cached steps: {reduced_steps}")

                # latents = torch.stack(latents, dim=1) 
                # latents_accelerated = torch.stack(latents_accelerated, dim=1)
                # timesteps = pipeline.scheduler.timesteps.repeat(len(prompts), 1).to(device) 

                rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
                rewards_future_cache = executor.submit(reward_fn, images_cache, prompts, prompt_metadata, only_strict=True)
                score_details, _ = rewards_future.result()
                score_details_cache, _ = rewards_future_cache.result()
                time.sleep(0)

                # ==================================================== Main part for acceleration policy update ====================================================
                # print(score_details)
                # score_details: a dict containing all sub-tasks of certain metrics
                # 'avg' denotes a weighted sum of all metrics when multiple metrics are considered
                # here because we only evaluate on Geneval, so 'avg' is equivalent
                R_full = torch.tensor(score_details["avg"], device=device).detach()
                R_cache = torch.tensor(score_details_cache["avg"], device=device).detach()
                R_drop = R_cache - R_full

                speedup = 1 - float(num_full_steps) / float(config.sample.num_steps)
                R_speedup = torch.full((R_cache.shape[0],), speedup, device=device, dtype=R_cache.dtype)

                R_final = config.alpha * R_cache + config.beta * R_speedup * R_drop - config.gamma * R_speedup * torch.relu(-R_drop)

                # TODO: Apply scaler
                acceleration_policy_log_probss_sum = acceleration_policy_log_probs.sum(dim=0)
                advantage_acceleration = (R_final - R_final.mean()) / (R_final.std() + 1e-4)
                loss_policy_acceleration = -(advantage_acceleration.detach() * acceleration_policy_log_probss_sum).mean()

                optimizer.zero_grad()
                loss_policy_acceleration.backward()
                optimizer.step()

                if is_main_process(rank):
                    wandb.log(
                        {
                            "train/[Quality]R_full": R_full.mean().item(),
                            "train/[Quality]R_cache": R_cache.mean().item(),
                            "train/[Speedup]R_speedup": (1 - float(num_full_steps) / float(config.sample.num_steps)),
                            "train/R_final": R_final.mean().item(),
                            "train/advantage": advantage_acceleration.detach().mean().item(),
                            "train/log_prob": acceleration_policy_log_probss_sum.mean().item(),
                            "train/loss_policy": loss_policy_acceleration.item(),
                        },
                        step=global_step,
                    )

            else:
                with torch.enable_grad():
                    _, _, _, sft_loss = pipeline_with_logprob_layerwise_cache_batch(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds[: len(prompts)],
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds[: len(prompts)],
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=config.sample.noise_level,
                        deterministic=config.sample.deterministic,
                        solver=config.sample.solver,
                        model_type="sd3",
                        acceleration_policy=acceleration_policy_ddp.module,
                        actions=config.ACTIONS,
                        skipped_layers = None,
                        sft = config.sft,
                    )

                optimizer.zero_grad()
                sft_loss.backward()
                optimizer.step()

                if is_main_process(rank):
                    wandb.log(
                        {
                            "train/sft_loss": sft_loss.item(),
                        },
                        step=global_step,
                    )
                
                if global_step > 150 and global_step % config.sft_eval_interval == 0:
                    with torch.enable_grad():
                        # inference with policy with cache
                        _, _, _, acceleration_policy_log_probs, num_full_steps, full_compute_steps = pipeline_with_logprob_layerwise_cache_batch(
                            pipeline,
                            prompt_embeds=prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            negative_prompt_embeds=sample_neg_prompt_embeds[: len(prompts)],
                            negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds[: len(prompts)],
                            num_inference_steps=config.sample.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                            output_type="pt",
                            height=config.resolution,
                            width=config.resolution,
                            noise_level=config.sample.noise_level,
                            deterministic=config.sample.deterministic,
                            solver=config.sample.solver,
                            model_type="sd3",
                            acceleration_policy=acceleration_policy_ddp.module,
                            actions=config.ACTIONS,
                            skipped_layers = None,
                            sft = False,
                        )
                        if is_main_process(rank):
                            save_dir = "full_compute_log_sft"
                            os.makedirs(save_dir, exist_ok=True)  

                            save_path = os.path.join(save_dir, f"{global_step}.json") 

                            with open(save_path, "w") as f:
                                json.dump(full_compute_steps, f, indent=2)
                            
                            save_policy(acceleration_policy_ddp.module, f"ckpt_policy/{global_step}.pt")


                    # with torch.no_grad():
                    #     _, _, _, sft_loss_eval = pipeline_with_logprob_layerwise_cache_batch(
                    #         pipeline,
                    #         prompt_embeds=prompt_embeds,
                    #         pooled_prompt_embeds=pooled_prompt_embeds,
                    #         negative_prompt_embeds=sample_neg_prompt_embeds[: len(prompts)],
                    #         negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds[: len(prompts)],
                    #         num_inference_steps=config.sample.num_steps,
                    #         guidance_scale=config.sample.guidance_scale,
                    #         output_type="pt",
                    #         height=config.resolution,
                    #         width=config.resolution,
                    #         noise_level=config.sample.noise_level,
                    #         deterministic=config.sample.deterministic,
                    #         solver=config.sample.solver,
                    #         model_type="sd3",
                    #         acceleration_policy=acceleration_policy_ddp.module,
                    #         actions=config.ACTIONS,
                    #         skipped_layers=None,
                    #         sft=True,   # 用 teacher loss 做评估
                    #     )

                    # # DDP：把各 rank 的 loss 做平均，避免某个 rank 的偶然 batch 触发停止
                    # loss_tensor = torch.tensor([float(sft_loss_eval.item())], device="cuda")
                    # dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    # loss_mean = (loss_tensor / dist.get_world_size()).item()

                    # # 置信度（几何平均意义上的 teacher action prob）
                    # conf = math.exp(-loss_mean)

                    # if is_main_process(rank):
                    #     wandb.log(
                    #         {
                    #             "eval/sft_loss": loss_mean,
                    #             "eval/teacher_conf": conf,
                    #         },
                    #         step=global_step,
                    #     )

                    # # patience 逻辑
                    # if is_main_process(rank):
                    #     logger.info(f'sft loss at {global_step} is {loss_mean}')
                    #     logger.info(f'threhold is {threshold}')
                    
                    # if loss_mean <= threshold:
                    #     sft_stop_patience += 1
                    #     if is_main_process(rank):
                    #         logger.info(f'patience increased to {sft_stop_patience}')
                    # else:
                    #     sft_stop_patience = 0
                    #     if is_main_process(rank):
                    #         logger.info(f'patience reset')

                    # stop_now = (sft_stop_patience >= 3)

                    # # 广播 stop flag，让所有 rank 同步退出
                    # stop_flag = torch.tensor([1 if stop_now else 0], device="cuda", dtype=torch.int)
                    # dist.broadcast(stop_flag, src=0)
                    # stop_now = bool(stop_flag.item())

                    # if stop_now:
                    #     should_stop = True
                    #     if is_main_process(rank):
                    #         save_policy(acceleration_policy_ddp.module, f"ckpt_policy/0.99_{global_step}.pt")
                    #     break
                        

            # if should_stop:
            #     break
            global_step += 1

            if ema is not None:
                ema.step(policy_trainable_parameters, global_step)

        # if should_stop:
        #     break








        # =====================================================之后要同时训练policy和diffusion的话再用=====================================================
        #     samples_data_list.append(
        #         {
        #             "prompt_ids": prompt_ids,
        #             "prompt_embeds": prompt_embeds,
        #             "pooled_prompt_embeds": pooled_prompt_embeds,
        #             "timesteps": timesteps,
        #             "next_timesteps": torch.concatenate([timesteps[:, 1:], torch.zeros_like(timesteps[:, :1])], dim=1),
        #             "latents_clean": latents[:, -1], # 预测目标
        #             "rewards_future": rewards_future,  # Store future
        #         }
        #     )

        #     collated_samples = {
        #         k: (
        #             torch.cat([s[k] for s in samples_data_list], dim=0)
        #             if not isinstance(samples_data_list[0][k], dict)
        #             else {sk: torch.cat([s[k][sk] for s in samples_data_list], dim=0) for sk in samples_data_list[0][k]}
        #         )
        #         for k in samples_data_list[0].keys()
        #     }

        # # Gather rewards across processes
        # gathered_rewards_dict = {}
        # for key, value_tensor in collated_samples["rewards"].items():
        #     gathered_rewards_dict[key] = gather_tensor_to_all(value_tensor, world_size).numpy()

        # if is_main_process(rank):  # logging
        #     wandb.log(
        #         {
        #             "epoch": epoch,
        #             **{
        #                 f"reward_{k}": v.mean()
        #                 for k, v in gathered_rewards_dict.items()
        #                 if "_strict_accuracy" not in k and "_accuracy" not in k
        #             },
        #         },
        #         step=global_step,
        #     )

        # avg_rewards_all = gathered_rewards_dict["avg"]
        
        # advantages = (avg_rewards_all - avg_rewards_all.mean()) / (avg_rewards_all.std() + 1e-4)
        # # Distribute advantages back to processes
        # samples_per_gpu = collated_samples["timesteps"].shape[0]
        # if advantages.ndim == 1:
        #     advantages = advantages[:, None]

        # if advantages.shape[0] == world_size * samples_per_gpu:
        #     collated_samples["advantages"] = torch.from_numpy(
        #         advantages.reshape(world_size, samples_per_gpu, -1)[rank]
        #     ).to(device)
        # else:
        #     assert False

        # if is_main_process(rank):
        #     logger.info(f"Advantages mean: {collated_samples['advantages'].abs().mean().item()}")

        # del collated_samples["rewards"]
        # del collated_samples["prompt_ids"]

        if world_size > 1:
            dist.barrier()


    if is_main_process(rank):
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    app.run(main)
