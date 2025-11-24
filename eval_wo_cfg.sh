# Local LoRA checkpoint, w/o CFG
export PYTHONPATH="$(pwd):$PYTHONPATH"
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 scripts/evaluation.py \
    --checkpoint_path "logs/nft/sd3/geneval/checkpoints/checkpoint-1018" \
    --model_type sd3 \
    --dataset geneval \
    --guidance_scale 1.0 \
    --mixed_precision fp16 \
    --save_images