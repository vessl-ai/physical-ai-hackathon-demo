#!/usr/bin/env python3
"""
LeRobot Dataset Visualization Dashboard
VESSL AI - Physical AI Hackathon Demo

Web-based visualization for robot manipulation datasets.
"""

import argparse
from pathlib import Path
from typing import Optional, List
import json

import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from PIL import Image

# Try to import optional dependencies
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class DatasetVisualizer:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.dataset = None
        self.episodes = []
        self.stats = {}
        self.image_keys = []

    def load_dataset(self):
        """Load dataset from HuggingFace or local path."""
        if not HAS_DATASETS:
            return "Error: 'datasets' library not installed. Run: pip install datasets"

        try:
            if Path(self.dataset_name).exists():
                # Local dataset
                self.dataset = load_dataset("parquet", data_dir=self.dataset_name, split="train")
            else:
                # HuggingFace dataset
                self.dataset = load_dataset(self.dataset_name, split="train")

            self._compute_stats()
            self._find_image_keys()
            return f"Loaded {len(self.dataset)} frames from {self.dataset_name}"
        except Exception as e:
            return f"Error loading dataset: {e}"

    def _find_image_keys(self):
        """Find all image-related keys in the dataset."""
        self.image_keys = []
        if self.dataset is None:
            return
        
        for key in self.dataset.features.keys():
            feature = self.dataset.features[key]
            # Check if it's an Image feature
            if hasattr(feature, 'decode') or 'Image' in str(type(feature)):
                self.image_keys.append(key)
            # Also check for common image key patterns
            elif any(pattern in key.lower() for pattern in ['image', 'rgb', 'camera', 'observation.images']):
                self.image_keys.append(key)
        
        # Remove duplicates
        self.image_keys = list(set(self.image_keys))
        print(f"Found image keys: {self.image_keys}")

    def _compute_stats(self):
        """Compute dataset statistics."""
        if self.dataset is None:
            return

        # Get unique episodes
        if "episode_index" in self.dataset.features:
            episode_indices = self.dataset["episode_index"]
            unique_episodes = list(set(episode_indices))
            self.stats["num_episodes"] = len(unique_episodes)
            self.stats["num_frames"] = len(self.dataset)
            self.stats["avg_episode_length"] = len(self.dataset) / len(unique_episodes)
        else:
            self.stats["num_episodes"] = 1
            self.stats["num_frames"] = len(self.dataset)
            self.stats["avg_episode_length"] = len(self.dataset)

        # Get feature info
        self.stats["features"] = list(self.dataset.features.keys())

    def get_stats_text(self) -> str:
        """Get formatted statistics text."""
        if not self.stats:
            return "No dataset loaded"

        image_keys_str = ", ".join(self.image_keys) if self.image_keys else "None found"
        
        text = f"""
## Dataset Statistics

- **Dataset**: {self.dataset_name}
- **Total Episodes**: {self.stats.get('num_episodes', 'N/A')}
- **Total Frames**: {self.stats.get('num_frames', 'N/A')}
- **Avg Episode Length**: {self.stats.get('avg_episode_length', 0):.1f} frames

### Image Keys
{image_keys_str}

### All Features
{', '.join(self.stats.get('features', []))}
"""
        return text

    def plot_joint_trajectory(self, episode_idx: int = 0) -> plt.Figure:
        """Plot joint trajectory for an episode."""
        if self.dataset is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No dataset loaded", ha='center', va='center')
            return fig

        # Filter to specific episode
        if "episode_index" in self.dataset.features:
            episode_data = self.dataset.filter(lambda x: x["episode_index"] == episode_idx)
        else:
            episode_data = self.dataset

        if len(episode_data) == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Episode {episode_idx} not found", ha='center', va='center')
            return fig

        # Find state column
        state_col = None
        for col in ["observation.state", "state", "robot_state", "qpos"]:
            if col in episode_data.features:
                state_col = col
                break

        if state_col is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No state data found", ha='center', va='center')
            return fig

        # Get states
        states = np.array(episode_data[state_col])

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot joint positions
        ax1 = axes[0]
        num_joints = min(states.shape[1], 7)  # ALOHA has 7 joints per arm
        for i in range(num_joints):
            ax1.plot(states[:, i], label=f"Joint {i+1}")
        ax1.set_xlabel("Timestep")
        ax1.set_ylabel("Joint Position")
        ax1.set_title(f"Episode {episode_idx} - Joint Trajectories (First {num_joints} joints)")
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Plot joint velocities (finite difference)
        ax2 = axes[1]
        velocities = np.diff(states, axis=0)
        for i in range(num_joints):
            ax2.plot(velocities[:, i], label=f"Joint {i+1}")
        ax2.set_xlabel("Timestep")
        ax2.set_ylabel("Joint Velocity")
        ax2.set_title("Joint Velocities (Finite Difference)")
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_action_distribution(self) -> plt.Figure:
        """Plot action distribution histogram."""
        if self.dataset is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No dataset loaded", ha='center', va='center')
            return fig

        # Find action column
        action_col = None
        for col in ["action", "actions", "robot_action"]:
            if col in self.dataset.features:
                action_col = col
                break

        if action_col is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No action data found", ha='center', va='center')
            return fig

        actions = np.array(self.dataset[action_col])
        n_dims = min(actions.shape[1], 6)

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()

        for i in range(n_dims):
            ax = axes[i]
            ax.hist(actions[:, i], bins=50, alpha=0.7, color=f'C{i}')
            ax.set_xlabel(f"Action Dim {i+1}")
            ax.set_ylabel("Frequency")
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)

            # Add stats
            mean = np.mean(actions[:, i])
            std = np.std(actions[:, i])
            ax.set_title(f"Mean: {mean:.3f}, Std: {std:.3f}")

        plt.suptitle("Action Distribution per Dimension")
        plt.tight_layout()
        return fig

    def get_frame_image(self, frame_idx: int = 0):
        """Get image from a specific frame."""
        if self.dataset is None:
            return None

        frame_idx = int(frame_idx)
        if frame_idx >= len(self.dataset):
            frame_idx = len(self.dataset) - 1
        if frame_idx < 0:
            frame_idx = 0

        # Try each image key
        img_keys_to_try = self.image_keys + [
            "observation.images.top",
            "observation.image", 
            "image", 
            "rgb",
            "observation.images.wrist",
            "observation.images.side",
        ]
        
        for img_col in img_keys_to_try:
            if img_col not in self.dataset.features:
                continue
                
            try:
                sample = self.dataset[frame_idx]
                img = sample[img_col]
                
                # Handle different image formats
                if isinstance(img, Image.Image):
                    # Already a PIL Image
                    return img
                elif isinstance(img, np.ndarray):
                    # Numpy array
                    if img.dtype != np.uint8:
                        img = (img * 255).astype(np.uint8)
                    return Image.fromarray(img)
                elif isinstance(img, dict):
                    # Dict format (might have bytes or path)
                    if "bytes" in img:
                        import io
                        return Image.open(io.BytesIO(img["bytes"]))
                    elif "path" in img:
                        return Image.open(img["path"])
                elif hasattr(img, 'numpy'):
                    # Tensor-like
                    arr = img.numpy()
                    if arr.dtype != np.uint8:
                        arr = (arr * 255).astype(np.uint8)
                    return Image.fromarray(arr)
                    
            except Exception as e:
                print(f"Error loading image from {img_col}: {e}")
                continue

        # If no image found, return a placeholder
        print(f"No image found for frame {frame_idx}")
        return self._create_placeholder_image(frame_idx)
    
    def _create_placeholder_image(self, frame_idx: int) -> Image.Image:
        """Create a placeholder image when no image is available."""
        img = Image.new('RGB', (640, 480), color=(50, 50, 50))
        return img

    def get_episode_frames(self, episode_idx: int, num_frames: int = 8) -> List[Image.Image]:
        """Get multiple frames from an episode."""
        if self.dataset is None:
            return []

        episode_idx = int(episode_idx)
        
        # Find frames for this episode
        if "episode_index" in self.dataset.features:
            indices = [i for i, ep in enumerate(self.dataset["episode_index"]) if ep == episode_idx]
        else:
            indices = list(range(len(self.dataset)))

        if not indices:
            return []

        # Sample evenly spaced frames
        step = max(1, len(indices) // num_frames)
        selected_indices = indices[::step][:num_frames]

        frames = []
        for idx in selected_indices:
            img = self.get_frame_image(idx)
            if img is not None:
                frames.append(img)

        return frames


def create_dashboard(dataset_name: str):
    """Create Gradio dashboard."""
    visualizer = DatasetVisualizer(dataset_name)

    with gr.Blocks(title="LeRobot Dataset Visualizer - VESSL AI") as demo:
        gr.Markdown("# LeRobot Dataset Visualization Dashboard")
        gr.Markdown("**Powered by VESSL AI** | Physical AI Hackathon")

        with gr.Row():
            dataset_input = gr.Textbox(
                value=dataset_name,
                label="Dataset Name (HuggingFace ID or local path)"
            )
            load_btn = gr.Button("Load Dataset", variant="primary")

        load_status = gr.Textbox(label="Status", interactive=False)

        with gr.Tabs():
            with gr.Tab("Overview"):
                stats_md = gr.Markdown("No dataset loaded")
                refresh_stats_btn = gr.Button("Refresh Stats")

            with gr.Tab("Episode Viewer"):
                with gr.Row():
                    episode_slider = gr.Slider(
                        minimum=0, maximum=100, step=1, value=0,
                        label="Episode Index"
                    )
                    plot_btn = gr.Button("Plot Trajectory")

                trajectory_plot = gr.Plot(label="Joint Trajectory")

            with gr.Tab("Action Analysis"):
                action_plot_btn = gr.Button("Plot Action Distribution")
                action_plot = gr.Plot(label="Action Distribution")

            with gr.Tab("Frame Viewer"):
                gr.Markdown("### Single Frame View")
                with gr.Row():
                    frame_slider = gr.Slider(
                        minimum=0, maximum=1000, step=1, value=0,
                        label="Frame Index"
                    )
                    show_frame_btn = gr.Button("Show Frame")
                frame_image = gr.Image(label="Camera View", type="pil")
                
                gr.Markdown("### Episode Gallery")
                with gr.Row():
                    gallery_episode_slider = gr.Slider(
                        minimum=0, maximum=100, step=1, value=0,
                        label="Episode Index for Gallery"
                    )
                    show_gallery_btn = gr.Button("Show Episode Frames")
                frame_gallery = gr.Gallery(label="Episode Frames", columns=4, rows=2, height="auto")

        # Event handlers
        def load_and_update(ds_name):
            visualizer.dataset_name = ds_name
            status = visualizer.load_dataset()
            stats = visualizer.get_stats_text()
            max_ep = max(0, visualizer.stats.get("num_episodes", 1) - 1)
            max_frame = max(0, visualizer.stats.get("num_frames", 1) - 1)
            return (
                status, 
                stats, 
                gr.Slider(maximum=max_ep), 
                gr.Slider(maximum=max_frame),
                gr.Slider(maximum=max_ep)
            )

        load_btn.click(
            load_and_update,
            inputs=[dataset_input],
            outputs=[load_status, stats_md, episode_slider, frame_slider, gallery_episode_slider]
        )

        refresh_stats_btn.click(
            lambda: visualizer.get_stats_text(),
            outputs=[stats_md]
        )

        plot_btn.click(
            visualizer.plot_joint_trajectory,
            inputs=[episode_slider],
            outputs=[trajectory_plot]
        )

        action_plot_btn.click(
            visualizer.plot_action_distribution,
            outputs=[action_plot]
        )

        show_frame_btn.click(
            visualizer.get_frame_image,
            inputs=[frame_slider],
            outputs=[frame_image]
        )
        
        show_gallery_btn.click(
            visualizer.get_episode_frames,
            inputs=[gallery_episode_slider],
            outputs=[frame_gallery]
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="LeRobot Dataset Visualizer")
    parser.add_argument(
        "--dataset",
        default="lerobot/aloha_sim_transfer_cube_human_image",
        help="Dataset name or path"
    )
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    demo = create_dashboard(args.dataset)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
