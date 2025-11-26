export PYTHONPATH="$(pwd):$PYTHONPATH"
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train_rl_sd3.py \
  --config config/nft.py:sd3_geneval