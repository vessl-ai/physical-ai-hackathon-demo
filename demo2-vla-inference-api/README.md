# Demo 2: VLA Model Inference (VESSL AI)

Test trained VLA (Vision-Language-Action) models using LeRobot.
**No physical robot required** - test with dataset samples or synthetic images.

---

## Quick Start

### Test ACT Model with Dataset Samples

```bash
cd ~/physical-ai-demos/demo2-vla-inference-api
source ~/vessl-hackathon-env/bin/activate

# Test ACT model with actual dataset images
python test_vla_inference.py \
  --checkpoint ~/physical-ai-demos/demo1-solo-cli/outputs/train/lerobot_aloha_sim_transfer_cube_human_act/checkpoints/last/pretrained_model \
  --dataset lerobot/aloha_sim_transfer_cube_human \
  --num-samples 5

# Or test with synthetic data
python test_vla_inference.py \
  --checkpoint ~/physical-ai-demos/demo1-solo-cli/outputs/train/lerobot_aloha_sim_transfer_cube_human_act/checkpoints/last/pretrained_model \
  --num-steps 5
```

### Expected Output

```
============================================================
INFERENCE RESULTS
============================================================

Step 0:
  Inference time: 337.04ms
  Action (14 dims):
    Left arm:      [-0.0111, -0.2717, +0.1595, -0.1346, +0.3853, -0.2276]
    Left gripper:  -0.2639
    Right arm:     [-0.0778, -0.6308, +0.4720, +0.3003, +0.1841, +0.0612]
    Right gripper: +0.4759
...

============================================================
Average inference time: 67.67ms (14.8 Hz)
============================================================
```

---

## Scripts

### 1. test_vla_inference.py

Test VLA model inference with synthetic or real images.

```bash
# Basic test with synthetic data
python test_vla_inference.py --checkpoint <path>

# Test with real image
python test_vla_inference.py --checkpoint <path> --image scene.jpg

# Run more steps
python test_vla_inference.py --checkpoint <path> --num-steps 20

# Save results to JSON
python test_vla_inference.py --checkpoint <path> --output results.json

# Use CPU (if GPU memory issues)
python test_vla_inference.py --checkpoint <path> --device cpu
```

### 2. visualize_actions.py

Visualize predicted action sequences as plots.

```bash
python visualize_actions.py \
  --checkpoint <path> \
  --num-steps 50 \
  --output action_plot.png
```

---

## Understanding the Output

### ALOHA Action Format (14 dimensions)

| Index | Description |
|-------|-------------|
| 0-5 | Left arm joint positions (6 joints) |
| 6 | Left gripper (open/close) |
| 7-12 | Right arm joint positions (6 joints) |
| 13 | Right gripper (open/close) |

### Single Arm Format (6 dimensions)

| Index | Description |
|-------|-------------|
| 0-4 | Arm joint positions (5 joints) |
| 5 | Gripper (open/close) |

---

## Using with Solo CLI

For testing on real robot hardware:

```bash
# Interactive inference (requires robot)
solo robo --inference

# When prompted:
# Policy path: ~/physical-ai-demos/demo1-solo-cli/outputs/train/.../checkpoints/last/pretrained_model
```

---

## Checkpoint Locations

After training with `solo robo --train`:

```
outputs/train/<dataset>_<policy>/
├── checkpoints/
│   ├── 001000/pretrained_model/    # Step 1000
│   ├── 002000/pretrained_model/    # Step 2000
│   ├── ...
│   └── last -> 020000              # Symlink to latest
└── config.yaml
```

### Load Specific Checkpoint

```bash
# Latest (recommended)
--checkpoint .../checkpoints/last/pretrained_model

# Specific step
--checkpoint .../checkpoints/010000/pretrained_model
```

---

## Performance (H100)

| Metric | Value |
|--------|-------|
| First inference | ~300ms (model warmup) |
| Subsequent inference | ~0.3-0.5ms |
| Average (5 steps) | ~68ms |
| Throughput | ~15 Hz |

---

## Troubleshooting

### "No module named lerobot.policies"

```bash
pip install lerobot --upgrade
```

### "CUDA out of memory"

```bash
python test_vla_inference.py --checkpoint <path> --device cpu
```

### "FileNotFoundError: checkpoint"

Check the path exists:
```bash
ls -la ~/physical-ai-demos/demo1-solo-cli/outputs/train/
```
