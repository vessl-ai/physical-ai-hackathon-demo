# Demo 3: ALOHA Simulation (VESSL AI)

Web-based simulation for testing ACT policies on ALOHA Transfer Cube task.
**No physical robot required** - runs entirely in gym-aloha simulation.

## Overview

Test and validate ACT policies trained on `lerobot/aloha_sim_transfer_cube_human`
dataset in a simulated environment. Compare policy performance against random actions.

## Features

- **gym-aloha Integration**: AlohaTransferCube-v0 environment
- **ACT Policy Support**: Load pre-trained models from HuggingFace or local checkpoints
- **Real-time Visualization**: 30 FPS WebSocket streaming to browser
- **Performance Metrics**: Track success rate, rewards, and episode statistics
- **Mode Comparison**: Switch between random actions and policy inference

## Quick Start

### 1. Install Dependencies

```bash
cd ~/physical-ai-demos/demo3-simulation
pip install -r requirements.txt
```

### 2. Start Simulation Server

```bash
# Random actions only
python sim_server.py --port 8080

# With pre-trained ACT policy from HuggingFace
python sim_server.py --port 8080 \
    --checkpoint lerobot/act_aloha_sim_transfer_cube_human

# With local checkpoint
python sim_server.py --port 8080 \
    --checkpoint /path/to/best_model
```

### 3. Open Browser

```
http://localhost:8080
```

## Web Interface

The browser interface provides:

- **Live Video Stream**: 30 FPS simulation rendering
- **Mode Selector**: Switch between Random and Policy modes
- **Run Episode**: Execute a complete episode (400 steps max)
- **Statistics Panel**:
  - Episode count
  - Success rate
  - Last reward
  - Frame count
- **Policy Status**: Shows loaded policy name

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--port` | 8080 | Server port |
| `--checkpoint` | None | ACT policy checkpoint (HuggingFace ID or local path) |
| `--device` | cuda | Device for inference (cuda/cpu) |
| `--obs-type` | pixels_agent_pos | Observation type |

## API Endpoints

### GET /
Web interface (HTML page)

### GET /api/status
```json
{
  "env_name": "AlohaTransferCube-v0",
  "running": true,
  "mode": "policy",
  "frame_count": 1523,
  "episode_count": 5,
  "success_count": 4,
  "policy_loaded": true,
  "policy_name": "lerobot/act_aloha_sim_transfer_cube_human"
}
```

### POST /api/run_episode
Run one complete episode and return results:
```json
{
  "steps": 287,
  "total_reward": 1.0,
  "success": true,
  "episode": 6,
  "mode": "policy"
}
```

### POST /api/load_policy
Load a trained policy checkpoint:
```bash
curl -X POST "http://localhost:8080/api/load_policy?checkpoint=lerobot/act_aloha_sim_transfer_cube_human"
```

### WebSocket /ws
Real-time frame streaming and control.

## Expected Performance

| Mode | Success Rate | Notes |
|------|--------------|-------|
| Random | ~0% | Baseline - random joint movements |
| ACT Policy | >80% | Trained on human demonstrations |

## Task Description

**AlohaTransferCube-v0**: The right arm picks up a cube from the table
and transfers it to the left gripper. Success is determined by the cube
being grasped by the left gripper.

- **Episode Length**: 400 steps max
- **Action Space**: 14-dim continuous (6 joints + 1 gripper per arm)
- **Observation**: Camera images + joint positions

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    VESSL AI H100                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  gym-aloha  │    │ ACT Policy  │    │  FastAPI    │  │
│  │ Simulation  │───▶│  Inference  │───▶│   Server    │  │
│  │  (MuJoCo)   │    │   (GPU)     │    │ (WebSocket) │  │
│  └─────────────┘    └─────────────┘    └──────┬──────┘  │
└───────────────────────────────────────────────┼─────────┘
                                                │
                                          WebSocket
                                                │
                                         ┌──────▼──────┐
                                         │   Browser   │
                                         │  (30 FPS)   │
                                         └─────────────┘
```

## Dataset

The ACT policy is trained on:
- **Dataset**: `lerobot/aloha_sim_transfer_cube_human`
- **Episodes**: 50 human teleoperation demonstrations
- **Task**: Transfer cube from right to left gripper

## Troubleshooting

### "gym-aloha not installed"
```bash
pip install gym-aloha
```

### "lerobot not installed"
```bash
pip install lerobot
```

### Policy loading fails
- Check if the checkpoint path/HuggingFace ID is correct
- Ensure you have internet access for HuggingFace downloads
- Try using `--device cpu` if GPU memory is insufficient

### Low FPS
- Reduce browser window size
- Close other GPU-intensive applications
- Check network latency for remote servers

## Integration with Training

1. Train ACT policy using Solo-CLI or custom training script
2. Save checkpoint to `outputs/best_model`
3. Test in this simulation:
   ```bash
   python sim_server.py --checkpoint outputs/best_model
   ```
4. Iterate until policy achieves high success rate
