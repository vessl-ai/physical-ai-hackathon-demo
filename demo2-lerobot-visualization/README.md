# Demo 2: LeRobot Dataset Visualization (VESSL AI)

Web-based visualization dashboard for LeRobot datasets.
**No physical robot required** - visualize datasets from HuggingFace.

---

## Quick Start

```bash
cd ~/physical-ai-demos/demo2-lerobot-visualization
source ~/vessl-hackathon-env/bin/activate
pip install -r requirements.txt

# Run with image dataset (recommended)
python dashboard.py --dataset lerobot/aloha_sim_transfer_cube_human_image --port 7860
```

Access at: http://<server-ip>:7860

---

## Important: Dataset Naming

LeRobot datasets come in two formats:

| Dataset Name | Has Images | Format |
|--------------|------------|--------|
| lerobot/aloha_sim_transfer_cube_human | Video (MP4) | Images stored as video files |
| lerobot/aloha_sim_transfer_cube_human_image | **Yes (PNG)** | Images stored directly |

**For visualization, use the _image suffix version** which stores images directly as PNG files.

---

## Features

### 1. Overview Tab
- Dataset statistics (episodes, frames)
- Available features list
- Image keys detection

### 2. Episode Viewer
- Joint trajectory visualization
- Joint velocity plots (finite difference)
- Episode-by-episode navigation

### 3. Action Analysis
- Action distribution histograms
- Per-dimension statistics (mean, std)

### 4. Frame Viewer
- **Single Frame View**: Browse individual frames
- **Episode Gallery**: View multiple frames from an episode in a grid

---

## Usage Examples

```bash
# Default dataset (aloha_sim_transfer_cube_human_image)
python dashboard.py

# Specify different dataset
python dashboard.py --dataset lerobot/aloha_sim_insertion_human_image

```

---

## Troubleshooting

### No image found / Black frames

Use the _image suffix dataset:
```bash
# Correct (has direct images)
--dataset lerobot/aloha_sim_transfer_cube_human_image

# May not show images (video-based)
--dataset lerobot/aloha_sim_transfer_cube_human
```
