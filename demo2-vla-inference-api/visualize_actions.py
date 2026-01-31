#!/usr/bin/env python3
"""
Action Visualization Script
VESSL AI - Physical AI Hackathon

Visualize predicted action sequences from VLA models.
Supports comparison with ground truth actions from datasets.
"""

import argparse
import os
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for headless servers
import matplotlib.pyplot as plt
from PIL import Image


def load_policy(checkpoint_path: str, policy_type: str, device: str):
    """Load policy based on type."""
    if policy_type == "act":
        try:
            from lerobot.policies.act.modeling_act import ACTPolicy
        except ImportError:
            from lerobot.common.policies.act.modeling_act import ACTPolicy
        policy = ACTPolicy.from_pretrained(checkpoint_path)
    elif policy_type == "smolvla":
        try:
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        except ImportError:
            from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    policy.to(device)
    policy.eval()
    return policy


def load_dataset_episode(
    dataset_name: str,
    episode_idx: int = 0,
    max_frames: Optional[int] = None
) -> Dict:
    """Load a full episode from dataset including video frames."""
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    import json
    
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    
    # Find episode boundaries
    episode_indices = []
    current_episode = -1
    
    for i in range(len(dataset)):
        ep_idx = dataset[i].get("episode_index", 0)
        if ep_idx != current_episode:
            episode_indices.append(i)
            current_episode = ep_idx
    
    print(f"Found {len(episode_indices)} episodes")
    
    if episode_idx >= len(episode_indices):
        episode_idx = 0
        print(f"Episode index out of range, using episode 0")
    
    # Get episode range
    start_idx = episode_indices[episode_idx]
    end_idx = episode_indices[episode_idx + 1] if episode_idx + 1 < len(episode_indices) else len(dataset)
    
    if max_frames:
        end_idx = min(end_idx, start_idx + max_frames)
    
    print(f"Loading episode {episode_idx}: frames {start_idx} to {end_idx} ({end_idx - start_idx} frames)")
    
    # Try to load video frames
    video_frames = {}
    try:
        import av
        
        video_file = hf_hub_download(
            repo_id=dataset_name,
            filename="videos/observation.images.top/chunk-000/file-000.mp4",
            repo_type="dataset"
        )
        print(f"Loading video: {video_file}")
        
        container = av.open(video_file)
        stream = container.streams.video[0]
        
        frame_count = 0
        target_range = range(start_idx, end_idx)
        
        for frame in container.decode(stream):
            if frame_count >= end_idx:
                break
            if frame_count >= start_idx:
                video_frames[frame_count] = frame.to_image()
            frame_count += 1
        
        container.close()
        print(f"Loaded {len(video_frames)} video frames")
        
    except Exception as e:
        print(f"Warning: Could not load video: {e}")
    
    # Extract data
    images = []
    states = []
    actions = []
    
    for i in range(start_idx, end_idx):
        sample = dataset[i]
        
        # Image
        img = video_frames.get(i)
        images.append(img)
        
        # State
        state = sample.get("observation.state")
        if state is not None:
            if isinstance(state, list):
                state = np.array(state)
        states.append(state)
        
        # Action
        action = sample.get("action")
        if action is not None:
            if isinstance(action, list):
                action = np.array(action)
        actions.append(action)
    
    return {
        "images": images,
        "states": np.array([s for s in states if s is not None]),
        "actions": np.array([a for a in actions if a is not None]),
        "episode_idx": episode_idx,
        "num_frames": len(images),
        "start_idx": start_idx
    }


def generate_action_sequence_from_dataset(
    policy,
    episode_data: Dict,
    device: str
) -> np.ndarray:
    """Generate predicted actions for each frame in episode."""
    actions = []
    
    with torch.no_grad():
        for i, img in enumerate(episode_data["images"]):
            # Convert image to tensor
            if img is not None:
                if isinstance(img, Image.Image):
                    img = img.resize((640, 480))
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
                else:
                    img_tensor = torch.rand(1, 3, 480, 640)
            else:
                img_tensor = torch.rand(1, 3, 480, 640)
            
            img_tensor = img_tensor.to(device)
            
            # Get state
            if i < len(episode_data["states"]):
                state = episode_data["states"][i]
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            else:
                state_tensor = torch.zeros(1, 14, device=device)
            
            obs = {
                "observation.images.top": img_tensor,
                "observation.state": state_tensor
            }
            
            output = policy.select_action(obs)
            
            if isinstance(output, torch.Tensor):
                action = output.cpu().numpy()
            else:
                action = output["action"].cpu().numpy()
            
            if len(action.shape) > 1:
                action = action[0]
            
            actions.append(action)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(episode_data['images'])} frames")
    
    return np.array(actions)


def generate_action_sequence(policy, num_steps: int, device: str):
    """Generate a sequence of actions with synthetic data."""
    actions = []

    obs = {
        "observation.images.top": torch.rand(1, 3, 480, 640, device=device),
        "observation.state": torch.zeros(1, 14, device=device)
    }

    with torch.no_grad():
        for _ in range(num_steps):
            output = policy.select_action(obs)

            if isinstance(output, torch.Tensor):
                action = output.cpu().numpy()
            else:
                action = output["action"].cpu().numpy()

            if len(action.shape) > 1:
                action = action[0]

            actions.append(action)
            obs["observation.state"] = torch.tensor(action, device=device).unsqueeze(0)

    return np.array(actions)


def plot_actions_comparison(
    predicted: np.ndarray,
    ground_truth: Optional[np.ndarray],
    output_path: str,
    title: str = "Action Sequence Comparison"
):
    """Plot predicted vs ground truth actions."""
    num_steps = len(predicted)
    action_dim = predicted.shape[1]
    
    if action_dim == 14:  # ALOHA
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Left arm
        ax = axes[0, 0]
        for i in range(6):
            ax.plot(predicted[:, i], label=f"Pred J{i+1}", linestyle='-', alpha=0.8)
            if ground_truth is not None:
                ax.plot(ground_truth[:, i], label=f"GT J{i+1}", linestyle='--', alpha=0.5)
        ax.set_title("Left Arm Joints")
        ax.set_xlabel("Step")
        ax.set_ylabel("Position")
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        
        # Left gripper
        ax = axes[0, 1]
        ax.plot(predicted[:, 6], 'b-', linewidth=2, label="Predicted")
        if ground_truth is not None:
            ax.plot(ground_truth[:, 6], 'b--', linewidth=2, alpha=0.5, label="Ground Truth")
        ax.set_title("Left Gripper")
        ax.set_xlabel("Step")
        ax.set_ylabel("Position")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Right arm
        ax = axes[1, 0]
        for i in range(6):
            ax.plot(predicted[:, 7+i], label=f"Pred J{i+1}", linestyle='-', alpha=0.8)
            if ground_truth is not None:
                ax.plot(ground_truth[:, 7+i], label=f"GT J{i+1}", linestyle='--', alpha=0.5)
        ax.set_title("Right Arm Joints")
        ax.set_xlabel("Step")
        ax.set_ylabel("Position")
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        
        # Right gripper
        ax = axes[1, 1]
        ax.plot(predicted[:, 13], 'r-', linewidth=2, label="Predicted")
        if ground_truth is not None:
            ax.plot(ground_truth[:, 13], 'r--', linewidth=2, alpha=0.5, label="Ground Truth")
        ax.set_title("Right Gripper")
        ax.set_xlabel("Step")
        ax.set_ylabel("Position")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    else:  # Generic
        fig, ax = plt.subplots(figsize=(14, 8))
        for i in range(min(action_dim, 8)):
            ax.plot(predicted[:, i], label=f"Pred Dim{i}", linestyle='-')
            if ground_truth is not None:
                ax.plot(ground_truth[:, i], label=f"GT Dim{i}", linestyle='--', alpha=0.5)
        ax.set_title("Action Dimensions")
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved plot to: {output_path}")


def plot_actions(actions: np.ndarray, output_path: str, title: str = "Action Sequence"):
    """Plot action sequence."""
    plot_actions_comparison(actions, None, output_path, title)


def plot_action_heatmap(
    predicted: np.ndarray,
    ground_truth: Optional[np.ndarray],
    output_path: str
):
    """Plot action heatmap with optional comparison."""
    if ground_truth is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        ax = axes[0]
        im = ax.imshow(predicted.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Action Dimension")
        ax.set_title("Predicted Actions")
        plt.colorbar(im, ax=ax, label="Value")
        
        ax = axes[1]
        im = ax.imshow(ground_truth.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Action Dimension")
        ax.set_title("Ground Truth Actions")
        plt.colorbar(im, ax=ax, label="Value")
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(predicted.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Action Dimension")
        ax.set_title("Action Sequence Heatmap")
        plt.colorbar(im, ax=ax, label="Value")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved heatmap to: {output_path}")


def plot_error_analysis(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    output_path: str
):
    """Plot error analysis between predicted and ground truth."""
    error = np.abs(predicted - ground_truth)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Error heatmap
    ax = axes[0, 0]
    im = ax.imshow(error.T, aspect="auto", cmap="Reds")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Action Dimension")
    ax.set_title("Absolute Error Heatmap")
    plt.colorbar(im, ax=ax, label="Error")
    
    # Mean error per timestep
    ax = axes[0, 1]
    mean_error_per_step = np.mean(error, axis=1)
    ax.plot(mean_error_per_step, 'b-', linewidth=1.5)
    ax.fill_between(range(len(mean_error_per_step)), mean_error_per_step, alpha=0.3)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("MAE Over Time")
    ax.grid(True, alpha=0.3)
    
    # Mean error per dimension
    ax = axes[1, 0]
    mean_error_per_dim = np.mean(error, axis=0)
    dim_labels = [
        "L_J1", "L_J2", "L_J3", "L_J4", "L_J5", "L_J6", "L_Grip",
        "R_J1", "R_J2", "R_J3", "R_J4", "R_J5", "R_J6", "R_Grip"
    ] if len(mean_error_per_dim) == 14 else [f"D{i}" for i in range(len(mean_error_per_dim))]
    
    bars = ax.bar(range(len(mean_error_per_dim)), mean_error_per_dim, color='steelblue')
    ax.set_xticks(range(len(mean_error_per_dim)))
    ax.set_xticklabels(dim_labels, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel("Action Dimension")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("MAE Per Dimension")
    ax.grid(True, alpha=0.3, axis='y')
    
    # Error distribution
    ax = axes[1, 1]
    ax.hist(error.flatten(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(error), color='red', linestyle='--', label=f"Mean: {np.mean(error):.4f}")
    ax.axvline(np.median(error), color='orange', linestyle='--', label=f"Median: {np.median(error):.4f}")
    ax.set_xlabel("Absolute Error")
    ax.set_ylabel("Frequency")
    ax.set_title("Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("Error Analysis: Predicted vs Ground Truth", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved error analysis to: {output_path}")
    print(f"  Overall MAE: {np.mean(error):.4f}")
    print(f"  Overall Median Error: {np.median(error):.4f}")
    print(f"  Max Error: {np.max(error):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Visualize VLA Actions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--policy-type", type=str, default="act", choices=["act", "smolvla"])
    parser.add_argument("--dataset", type=str, default=None, help="HuggingFace dataset name")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to visualize")
    parser.add_argument("--max-frames", type=int, default=200, help="Max frames to process")
    parser.add_argument("--num-steps", type=int, default=50, help="Number of steps (synthetic mode)")
    parser.add_argument("--output", type=str, default="action_visualization.png")
    parser.add_argument("--heatmap", action="store_true", help="Also generate heatmap")
    parser.add_argument("--error-analysis", action="store_true", help="Generate error analysis")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 50)
    print("Action Visualization - VESSL AI")
    print("=" * 50)

    # Load policy
    print(f"Loading {args.policy_type} policy...")
    policy = load_policy(args.checkpoint, args.policy_type, args.device)

    if args.dataset:
        # Load episode from dataset
        print(f"\nLoading episode {args.episode} from {args.dataset}...")
        episode_data = load_dataset_episode(args.dataset, args.episode, args.max_frames)
        
        # Generate predictions
        print(f"\nGenerating predictions for {episode_data['num_frames']} frames...")
        predicted = generate_action_sequence_from_dataset(policy, episode_data, args.device)
        ground_truth = episode_data["actions"]
        
        # Align lengths
        min_len = min(len(predicted), len(ground_truth))
        predicted = predicted[:min_len]
        ground_truth = ground_truth[:min_len]
        
        print(f"\nPredicted shape: {predicted.shape}")
        print(f"Ground truth shape: {ground_truth.shape}")
        
        # Plot comparison
        title = f"Episode {args.episode}: Predicted vs Ground Truth ({args.policy_type.upper()})"
        plot_actions_comparison(predicted, ground_truth, args.output, title)
        
        if args.heatmap:
            heatmap_path = args.output.replace(".png", "_heatmap.png")
            plot_action_heatmap(predicted, ground_truth, heatmap_path)
        
        if args.error_analysis:
            error_path = args.output.replace(".png", "_error.png")
            plot_error_analysis(predicted, ground_truth, error_path)
    else:
        # Synthetic mode
        print(f"\nGenerating {args.num_steps} actions (synthetic)...")
        actions = generate_action_sequence(policy, args.num_steps, args.device)
        print(f"Action shape: {actions.shape}")
        
        plot_actions(actions, args.output, f"Predicted Actions ({args.policy_type.upper()})")
        
        if args.heatmap:
            heatmap_path = args.output.replace(".png", "_heatmap.png")
            plot_action_heatmap(actions, None, heatmap_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
