#!/bin/bash
# Solo-CLI Training Demo Setup Script
# VESSL AI - Physical AI Hackathon
# This demo works WITHOUT physical robot hardware

set -e

echo "=============================================="
echo "Solo-CLI Training Demo Setup"
echo "VESSL AI - Physical AI Hackathon"
echo "=============================================="
echo ""
echo "This demo showcases policy training with"
echo "existing HuggingFace datasets (no robot needed)"
echo ""

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv
    echo ""
else
    echo "Warning: No NVIDIA GPU detected"
fi

# Create/activate virtual environment
if [ ! -d "$HOME/vessl-hackathon-env" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv ~/vessl-hackathon-env
fi
source ~/vessl-hackathon-env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA
echo ""
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Solo-CLI from source
echo ""
echo "Installing Solo-CLI..."
cd ~
if [ ! -d "solo-cli" ]; then
    git clone https://github.com/GetSoloTech/solo-cli.git
fi
cd solo-cli
git pull origin main
pip install -e .

# Install LeRobot dependencies
echo ""
echo "Installing LeRobot..."
pip install lerobot

# Install additional dependencies
pip install transformers accelerate datasets huggingface_hub

# Download sample dataset
echo ""
echo "Downloading sample dataset for demo..."
python3 -c "from datasets import load_dataset; ds = load_dataset('lerobot/aloha_sim_transfer_cube_human', split='train[:10]'); print(f'Sample loaded: {len(ds)} frames')"

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Activate environment:"
echo "  source ~/vessl-hackathon-env/bin/activate"
echo ""
echo "Start training demo:"
echo "  solo robo --train"
echo ""
echo "Available commands:"
echo "  solo robo --train      Train policy on HuggingFace dataset"
echo "  solo --help            Show all Solo-CLI commands"
echo ""
echo "Example datasets:"
echo "  lerobot/aloha_sim_transfer_cube_human"
echo "  lerobot/pusht"
echo "  lerobot/xarm_lift_medium"
echo ""
