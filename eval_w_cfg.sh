# Hugging Face LoRA checkpoint, w/ CFG
export PYTHONPATH="$(pwd):$PYTHONPATH"
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 scripts/evaluation.py \
    --lora_hf_path "jieliu/SD3.5M-FlowGRPO-GenEval" \
    --model_type sd3 \
    --dataset geneval \
    --guidance_scale 4.5 \
    --mixed_precision fp16 \
    --save_images