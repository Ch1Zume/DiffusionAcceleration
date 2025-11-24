export PYTHONPATH="$(pwd):$PYTHONPATH"
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train_nft_flux.py \
  --config config/nft.py:flux_geneval