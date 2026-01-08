import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from peft import PeftModel

model_id = "stabilityai/stable-diffusion-3.5-medium"
lora_ckpt_path = "jieliu/SD3.5M-FlowGRPO-GenEval"
device = "cuda"


pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_ckpt_path)
pipe.transformer = pipe.transformer.merge_and_unload()
pipe = pipe.to(device)

prompt = 'a red colored car.'
image = pipe(prompt, height=768, width=768, num_inference_steps=40,guidance_scale=4.5,negative_prompt="").images[0]  
image.save(f"flow_grpo.png")