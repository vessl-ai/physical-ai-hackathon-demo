# Demo 1: Solo-CLI Training Demo (VESSL AI)

This demo showcases training VLA (Vision-Language-Action) policies using Solo-CLI
with existing HuggingFace datasets. **No physical robot required.**

## What is Solo-CLI?

Solo-CLI is a command-line tool for Physical AI that provides:
- Robot control (calibration, teleoperation) - requires physical robot
- Data recording - requires physical robot
- **Policy training** - works with existing datasets (no robot needed)
- **Model serving** - deploy AI models locally (no robot needed)

This demo focuses on the **training** capabilities that work without hardware.

---

## Recommended Datasets for Physical AI Hackathon

Based on the hackathon robots (SO-101, Koch, LeKiwi, OpenDroid, Unitree G1),
we recommend the following datasets:

### Best Datasets for SO-101 / SO-100 Robot Arms

| Dataset | Task | Episodes | Recommended For |
|---------|------|----------|-----------------|
| **`lerobot/svla_so101_pickplace`** | Pick & Place | 50 | SmolVLA training |
| **`lerobot/svla_so100_pickplace`** | Pick & Place | 50 | SmolVLA training |
| `lerobot/svla_so100_stacking` | Block stacking | 50 | SmolVLA training |
| `lerobot/svla_so100_sorting` | Object sorting | 50 | SmolVLA training |

### Best Datasets for Koch Robot Arms

| Dataset | Task | Episodes | Recommended For |
|---------|------|----------|-----------------|
| **`lerobot/koch_pick_place_1_lego`** | Lego pick & place | 50 | ACT training |
| `lerobot/koch_pick_place_5_lego` | Multi-lego | 50 | ACT training |

### Best Datasets for General Training (Simulation)

| Dataset | Task | Episodes | Recommended For |
|---------|------|----------|-----------------|
| **`lerobot/aloha_sim_transfer_cube_human`** | Cube transfer | 50 | ACT, any policy |
| `lerobot/aloha_sim_insertion_human` | Peg insertion | 400 | ACT, any policy |
| `lerobot/pusht` | Push-T task | 200 | Diffusion Policy |
| `lerobot/xarm_lift_medium` | Object lifting | 100 | ACT |

### Our Top Recommendation

For this hackathon, we recommend:

```bash
# Best for SO-101 users (most hackathon participants)
lerobot/svla_so101_pickplace

# Best for quick demo (small, fast training)
lerobot/aloha_sim_transfer_cube_human
```

**Why `lerobot/svla_so101_pickplace`?**
- Matches SO-101 robot used at hackathon
- Optimized for SmolVLA model
- 50 episodes = fast training (~15 min on H100)
- Pick & place is fundamental manipulation task

---

## Supported Policy Types

| Policy | Description | Best Dataset |
|--------|-------------|--------------|
| **SmolVLA** | Lightweight VLA by HuggingFace | `svla_so101_pickplace` |
| **ACT** | Action Chunking Transformer | `aloha_sim_transfer_cube_human` |
| **Pi0** | Physical Intelligence model | Any dataset |
| **Diffusion** | Diffusion Policy | `pusht` |

## Quick Start

### 1. Install Solo-CLI

```bash
# Activate environment
source ~/vessl-hackathon-env/bin/activate

# Install from source (recommended)
cd ~/solo-cli
pip install -e .
```

### 2. Train with Recommended Dataset

```bash
# Interactive training
solo robo --train

# When prompted:
# Dataset: lerobot/svla_so101_pickplace
# Policy: SmolVLA
# Steps: 10000
# Batch size: 64
```

### 3. Or Use CLI Arguments

```bash
# Direct training command (skip prompts)
solo robo --train -y
```

## Training Example: SmolVLA on SO-101 Dataset

```bash
solo robo --train

# === Training Configuration ===
# Dataset: lerobot/svla_so101_pickplace
# Policy: SmolVLA (pretrained: lerobot/smolvla_base)
# Training steps: 10000
# Batch size: 64
# Output: outputs/smolvla_so101_pickplace
```

Expected output:
- Training time: ~15 minutes (H100)
- Checkpoint: `outputs/smolvla_so101_pickplace/checkpoint_10000/`

## Training Performance (VESSL AI H100)

| Dataset | Policy | Steps | Time |
|---------|--------|-------|------|
| svla_so101_pickplace | SmolVLA | 10k | ~15 min |
| aloha_sim_transfer_cube | ACT | 10k | ~10 min |
| pusht | Diffusion | 50k | ~30 min |
| koch_pick_place_1_lego | ACT | 10k | ~12 min |

## After Training

Once training is complete, you can:

1. **Visualize Dataset** (Demo 2)
   ```bash
   cd ../demo2-lerobot-visualization
   python dashboard.py --dataset lerobot/aloha_sim_transfer_cube_human_image --port 7860
   ```

2. **Test in Simulation** (Demo 3)
   ```bash
   cd ../demo3-simulation
   export MUJOCO_GL=egl
   python sim_server.py --port 8080 \
     --checkpoint outputs/train/lerobot_aloha_sim_transfer_cube_human_act/checkpoints/last/pretrained_model
   ```

3. **Deploy to Real Robot** (at hackathon with physical robot)
   ```bash
   solo robo --inference
   # Select your trained checkpoint
   ```

## Reference Links

- [Solo-CLI GitHub](https://github.com/GetSoloTech/solo-cli)
- [Solo-CLI Docs](https://docs.getsolo.tech)
- [LeRobot Datasets](https://huggingface.co/lerobot)
- [SmolVLA Model](https://huggingface.co/lerobot/smolvla_base)
- [SO-101 Documentation](https://huggingface.co/docs/lerobot/so101)
