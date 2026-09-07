# DiffusionAcceleration dependency modifications

The DiffusionAcceleration repository now keeps its project-specific dependency
code in the repository itself:

- `flow_grpo_patch/`: the existing project-specific Flow-GRPO snapshot,
  copied without Python caches, plus the locally modified
  `scripts/train_sd3.py`. It adds a small image-saving helper, loads the
  evaluation LoRA for a quick comparison, and keeps the transformer trainable
  for the local experiment.
- `taylorseer_patch/`: the existing project-specific TaylorSeer adapter,
  copied without Python caches. It provides the custom SD3 linear-attention
  integration used by the local RL scripts.
- `dependency_patches/DiffusionNFT/README.md`: preserves the local setup
  instructions that reuse already checked-out MMCV and MMDetection trees
  instead of cloning them during setup.
- `dependency_patches/flow_grpo/mmdetection/mmdet/__init__.py`: preserves the
  compatibility edit that raises the accepted MMCV upper bound to `2.3.0`.

The clean upstream repositories remain public dependencies. Checkpoints,
reward-model weights, logs, evaluation outputs, and generated test images were
not copied.
