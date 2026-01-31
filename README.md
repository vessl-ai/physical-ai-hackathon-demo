# Physical AI Demos

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LeRobot](https://img.shields.io/badge/LeRobot-HuggingFace-orange.svg)](https://github.com/huggingface/lerobot)

End-to-end demos for training and testing ACT (Action Chunking Transformer) policies on robot manipulation tasks.

> **No physical robot required** - All demos work with simulated environments and pre-recorded datasets.

## Overview

This repository demonstrates a complete Physical AI workflow using:
- **Dataset**: [`lerobot/aloha_sim_transfer_cube_human`](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human)
- **Policy**: ACT (Action Chunking Transformer)
- **Simulation**: [gym-aloha](https://github.com/huggingface/gym-aloha)
- **Task**: Transfer cube from right arm to left gripper (dual-arm ALOHA robot)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/physical-ai-demos.git
cd physical-ai-demos

# Run installation script
./install_all.sh

# Activate environment
source ./vessl-hackathon-env/bin/activate
```

### Run Simulation

```bash
cd demo4-simulation
export MUJOCO_GL=egl  # For headless rendering
python sim_server.py --port 8080
```

Open http://localhost:8080 in your browser.

## Demos

| Demo | Description | Command |
|------|-------------|---------|
| **demo1** | Train ACT policy | `solo robo --train` |
| **demo2** | Test inference | `python test_vla_inference.py` |
| **demo3** | Visualize dataset | `python dashboard.py` |
| **demo4** | Run simulation | `python sim_server.py` |

## Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   demo3: Visualize    demo1: Train    demo2: Test          │
│   ┌──────────┐       ┌──────────┐    ┌──────────┐          │
│   │ Dataset  │       │   ACT    │───▶│Inference │          │
│   │Dashboard │       │ Training │    │  Test    │          │
│   └──────────┘       └────┬─────┘    └──────────┘          │
│                           │                                 │
│                           ▼                                 │
│                    ┌──────────┐                             │
│                    │  demo4   │                             │
│                    │gym-aloha │                             │
│                    │Simulation│                             │
│                    └──────────┘                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Demo Details

### Demo 1: ACT Training (Solo-CLI)

Train ACT policy on ALOHA dataset using [Solo-CLI](https://github.com/GetSoloTech/solo-cli).

```bash
cd demo1-solo-cli
solo robo --train
# Select: lerobot/aloha_sim_transfer_cube_human
# Select: ACT
```

### Demo 2: Inference Test

Test trained ACT model with dataset samples.

```bash
cd demo2-vla-inference-api
python test_vla_inference.py \
  --checkpoint <path_to_checkpoint> \
  --dataset lerobot/aloha_sim_transfer_cube_human \
  --num-samples 5
```

### Demo 3: Dataset Visualization

Web dashboard for visualizing LeRobot datasets.

```bash
cd demo3-lerobot-visualization
python dashboard.py --dataset lerobot/aloha_sim_transfer_cube_human_image --port 7860
```

### Demo 4: ALOHA Simulation

Test ACT policy in gym-aloha simulation with real-time browser visualization.

```bash
cd demo4-simulation
export MUJOCO_GL=egl
python sim_server.py --port 8080 --checkpoint <path_to_checkpoint>
```

**Features:**
- Random vs Policy mode comparison
- Real-time 30 FPS streaming
- Success rate tracking
- Episode metrics (reward, steps)

## Dataset & Model

| Property | Value |
|----------|-------|
| Dataset | `lerobot/aloha_sim_transfer_cube_human` |
| Policy | ACT (Action Chunking Transformer) |
| Task | Transfer cube between dual arms |
| Action Space | 14-dim (6 joints + 1 gripper × 2 arms) |
| Observation | Camera image (480×640) + Joint positions (14-dim) |
| Episodes | 50 human demonstrations |

## Requirements

- Python 3.10+
- CUDA-compatible GPU (recommended: H100, A100, RTX 4090)
- 16GB+ GPU memory

### Key Dependencies

```
torch>=2.0.0
lerobot
gymnasium
gym-aloha
mujoco
fastapi
gradio
```

## Troubleshooting

### MuJoCo rendering fails (headless server)

```bash
export MUJOCO_GL=egl
# or
export MUJOCO_GL=osmesa
```

### CUDA out of memory

```bash
# Use CPU for inference
python sim_server.py --device cpu
```

### LeRobot import error

```bash
pip install lerobot --upgrade
```

## Project Structure

```
physical-ai-demos/
├── demo1-solo-cli/           # ACT training with Solo-CLI
│   ├── README.md
│   └── setup.sh
├── demo2-vla-inference-api/  # Inference testing
│   ├── test_vla_inference.py
│   └── requirements.txt
├── demo3-lerobot-visualization/  # Dataset dashboard
│   ├── dashboard.py
│   └── requirements.txt
├── demo4-simulation/         # gym-aloha simulation
│   ├── sim_server.py
│   └── requirements.txt
├── install_all.sh           # Full installation script
└── README.md
```

## References

- [LeRobot](https://github.com/huggingface/lerobot) - Robot learning framework
- [gym-aloha](https://github.com/huggingface/gym-aloha) - ALOHA simulation environment
- [ACT Paper](https://arxiv.org/abs/2304.13705) - Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware
- [Solo-CLI](https://github.com/GetSoloTech/solo-cli) - Physical AI CLI tool

## License

MIT License

## Acknowledgments

- [VESSL AI](https://vessl.ai) - MLOps Platform for Physical AI
- [HuggingFace](https://huggingface.co) - LeRobot and datasets
- [Physical Intelligence](https://www.physicalintelligence.company/) - ACT research
