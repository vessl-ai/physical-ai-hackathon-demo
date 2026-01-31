#!/bin/bash
# VESSL AI - Physical AI Hackathon
# Complete Installation Script

set -e

echo "========================================"
echo "VESSL AI - Physical AI Hackathon Setup"
echo "========================================"
echo ""

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv
    echo ""
else
    echo "Warning: No NVIDIA GPU detected"
fi

# Check and install pip if not available
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    echo "pip not found. Installing pip..."
    apt-get update && apt-get install -y python3-pip
fi

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv ./vessl-hackathon-env
source ./vessl-hackathon-env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA
echo ""
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Install common dependencies
echo ""
echo "Installing common dependencies..."
pip install \
    "transformers<5.0.0" \
    solo-cli \
    accelerate \
    datasets \
    huggingface_hub \
    numpy \
    pillow \
    opencv-python-headless \
    matplotlib \
    pandas \
    pyyaml \
    tqdm

# Install web server dependencies
echo ""
echo "Installing web server dependencies..."
pip install \
    fastapi \
    uvicorn \
    gradio \
    websockets \
    requests

# Install simulation dependencies
echo ""
echo "Installing simulation dependencies..."
pip install \
    gymnasium \
    mujoco \
    gym-aloha \
    einops

# Install LeRobot (for ACT policy)
echo ""
echo "Installing LeRobot..."
pip install lerobot || echo "LeRobot installation failed, some features may not work"

# Summary
echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "Activate environment:"
echo "  source ~/vessl-hackathon-env/bin/activate"
echo ""
echo "Demo directories:"
echo "  - demo1-solo-cli: Train ACT on aloha_sim_transfer_cube_human"
echo "  - demo2-lerobot-visualization: Dataset visualization"
echo "  - demo3-simulation: gym-aloha simulation with ACT policy"
echo ""
echo "Happy Hacking! - VESSL AI Team"
