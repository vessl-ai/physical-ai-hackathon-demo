#!/usr/bin/env python3
"""
VLA Model Inference Test Script
VESSL AI - Physical AI Hackathon

Test trained VLA models (ACT, SmolVLA, etc.) without physical robot.
Supports loading images from HuggingFace datasets (including video-based datasets).
"""

import argparse
import json
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import torch
from PIL import Image


def find_checkpoint_path(base_path: str) -> str:
    """Find the actual model checkpoint path."""
    base = Path(base_path)

    candidates = [
        base / "checkpoints" / "last" / "pretrained_model",
        base / "checkpoints" / "last",
        base / "pretrained_model",
        base / "checkpoint",
        base,
    ]

    for candidate in candidates:
        if candidate.exists():
            if (candidate / "config.json").exists() or (candidate / "model.safetensors").exists():
                return str(candidate)
            if list(candidate.glob("*.pt")) or list(candidate.glob("*.pth")):
                return str(candidate)

    return str(base)


def load_act_policy(checkpoint_path: str, device: str = "cuda"):
    """Load ACT policy from checkpoint."""
    try:
        from lerobot.policies.act.modeling_act import ACTPolicy
    except ImportError:
        from lerobot.common.policies.act.modeling_act import ACTPolicy

    actual_path = find_checkpoint_path(checkpoint_path)
    print(f"Loading ACT policy from: {actual_path}")

    policy = ACTPolicy.from_pretrained(actual_path)
    policy.to(device)
    policy.eval()

    return policy


def load_smolvla_policy(checkpoint_path: str, device: str = "cuda"):
    """Load SmolVLA policy from checkpoint."""
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except ImportError:
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    actual_path = find_checkpoint_path(checkpoint_path)
    print(f"Loading SmolVLA policy from: {actual_path}")

    policy = SmolVLAPolicy.from_pretrained(actual_path)
    policy.to(device)
    policy.eval()

    return policy


def load_dataset_with_videos(
    dataset_name: str = "lerobot/aloha_sim_transfer_cube_human",
    num_samples: int = 5,
    save_dir: Optional[str] = None
) -> List[Dict]:
    """Load samples from HuggingFace dataset, extracting frames from videos."""
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    import json
    
    print(f"Loading dataset: {dataset_name}")
    
    # Load parquet data for states and actions
    dataset = load_dataset(dataset_name, split="train")
    total_frames = len(dataset)
    print(f"Dataset loaded: {total_frames} frames")
    
    # Download and read info.json to get video path
    try:
        info_path = hf_hub_download(
            repo_id=dataset_name,
            filename="meta/info.json",
            repo_type="dataset"
        )
        with open(info_path) as f:
            info = json.load(f)
        
        video_path_template = info.get("video_path", "")
        has_video = "observation.images.top" in info.get("features", {})
        fps = info.get("fps", 50)
    except:
        has_video = False
        fps = 50
    
    # Select sample indices
    if num_samples >= total_frames:
        indices = list(range(total_frames))
    else:
        indices = [int(i * total_frames / num_samples) for i in range(num_samples)]
    
    samples = []
    
    # Try to load video frames
    video_frames = None
    if has_video:
        try:
            import av
            
            # Download video file
            video_file = hf_hub_download(
                repo_id=dataset_name,
                filename="videos/observation.images.top/chunk-000/file-000.mp4",
                repo_type="dataset"
            )
            print(f"Loading video: {video_file}")
            
            # Extract frames for requested indices
            container = av.open(video_file)
            stream = container.streams.video[0]
            
            video_frames = {}
            frame_count = 0
            target_indices = set(indices)
            
            for frame in container.decode(stream):
                if frame_count in target_indices:
                    img = frame.to_image()
                    video_frames[frame_count] = img
                frame_count += 1
                if len(video_frames) >= len(target_indices):
                    break
            
            container.close()
            print(f"Extracted {len(video_frames)} frames from video")
            
        except ImportError:
            print("Warning: PyAV not installed. Trying OpenCV...")
            try:
                import cv2
                video_file = hf_hub_download(
                    repo_id=dataset_name,
                    filename="videos/observation.images.top/chunk-000/file-000.mp4",
                    repo_type="dataset"
                )
                
                cap = cv2.VideoCapture(video_file)
                video_frames = {}
                frame_count = 0
                target_indices = set(indices)
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_count in target_indices:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video_frames[frame_count] = Image.fromarray(frame_rgb)
                    frame_count += 1
                    if len(video_frames) >= len(target_indices):
                        break
                
                cap.release()
                print(f"Extracted {len(video_frames)} frames using OpenCV")
                
            except Exception as e:
                print(f"Warning: Could not load video: {e}")
                video_frames = None
        except Exception as e:
            print(f"Warning: Could not load video: {e}")
            video_frames = None
    
    # Build samples
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        
        # Get image from video or create placeholder
        img = None
        if video_frames and idx in video_frames:
            img = video_frames[idx]
        
        # Get state
        state = sample.get("observation.state")
        if state is not None and isinstance(state, list):
            state = np.array(state)
        
        # Get action
        action = sample.get("action")
        if action is not None and isinstance(action, list):
            action = np.array(action)
        
        sample_data = {
            "index": idx,
            "image": img,
            "state": state,
            "action": action,
            "has_image": img is not None
        }
        samples.append(sample_data)
        
        # Save image if requested
        if save_dir and img is not None:
            os.makedirs(save_dir, exist_ok=True)
            img_path = os.path.join(save_dir, f"sample_{i:03d}_idx{idx}.png")
            img.save(img_path)
            print(f"  Saved: {img_path}")
    
    samples_with_images = sum(1 for s in samples if s["has_image"])
    print(f"Loaded {len(samples)} samples ({samples_with_images} with images)")
    return samples


def create_observation_from_sample(
    sample: Dict,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """Create observation dict from dataset sample."""
    img = sample.get("image")
    
    if img is not None:
        # Convert PIL Image to tensor
        if isinstance(img, Image.Image):
            img = img.resize((640, 480))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        else:
            img_tensor = torch.tensor(img).unsqueeze(0)
            if img_tensor.shape[1] != 3:
                img_tensor = img_tensor.permute(0, 3, 1, 2)
    else:
        # Create dummy image if no image available
        img_tensor = torch.rand(1, 3, 480, 640)
    
    img_tensor = img_tensor.to(device)
    
    # Handle state
    state = sample.get("state")
    if state is not None:
        if isinstance(state, (list, np.ndarray)):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        else:
            state_tensor = state.unsqueeze(0) if len(state.shape) == 1 else state
        state_tensor = state_tensor.to(device)
    else:
        state_tensor = torch.zeros(1, 14, device=device)
    
    obs = {
        "observation.images.top": img_tensor,
        "observation.state": state_tensor
    }
    
    return obs


def create_dummy_observation(
    image_size: tuple = (480, 640),
    state_dim: int = 14,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """Create dummy observation for testing."""
    obs = {
        "observation.images.top": torch.rand(1, 3, image_size[0], image_size[1], device=device),
        "observation.state": torch.zeros(1, state_dim, device=device)
    }
    return obs


def load_image_as_observation(
    image_path: str,
    state_dim: int = 14,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """Load real image as observation."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((640, 480))

    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor.to(device)

    obs = {
        "observation.images.top": img_tensor,
        "observation.state": torch.zeros(1, state_dim, device=device)
    }

    return obs


def run_inference(
    policy,
    observation: Dict[str, torch.Tensor],
    num_steps: int = 1
) -> List[Dict[str, Any]]:
    """Run inference and return results."""
    results = []

    with torch.no_grad():
        for step in range(num_steps):
            start_time = time.time()
            output = policy.select_action(observation)
            inference_time = (time.time() - start_time) * 1000

            if isinstance(output, dict):
                action = output.get("action", output.get("actions"))
                action_chunk = output.get("action_chunk")
            elif isinstance(output, torch.Tensor):
                action = output
                action_chunk = None
            else:
                raise ValueError(f"Unexpected output type: {type(output)}")

            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()

            if len(action.shape) > 1:
                action = action[0]

            result = {
                "step": step,
                "action": action.tolist(),
                "inference_time_ms": inference_time,
            }

            if action_chunk is not None:
                if isinstance(action_chunk, torch.Tensor):
                    action_chunk = action_chunk.cpu().numpy()
                result["action_chunk_shape"] = list(action_chunk.shape)

            results.append(result)

            if "observation.state" in observation:
                new_state = torch.tensor(action, device=observation["observation.state"].device).unsqueeze(0)
                observation["observation.state"] = new_state

    return results


def run_inference_on_samples(
    policy,
    samples: List[Dict],
    device: str = "cuda"
) -> List[Dict[str, Any]]:
    """Run inference on multiple dataset samples."""
    results = []
    
    with torch.no_grad():
        for i, sample in enumerate(samples):
            obs = create_observation_from_sample(sample, device)
            
            start_time = time.time()
            output = policy.select_action(obs)
            inference_time = (time.time() - start_time) * 1000
            
            if isinstance(output, torch.Tensor):
                action = output.cpu().numpy()
            else:
                action = output.get("action", output.get("actions"))
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
            
            if len(action.shape) > 1:
                action = action[0]
            
            # Get ground truth action
            gt_action = sample.get("action")
            if gt_action is not None:
                if hasattr(gt_action, 'tolist'):
                    gt_action = gt_action.tolist()
                elif isinstance(gt_action, np.ndarray):
                    gt_action = gt_action.tolist()
            
            result = {
                "sample_index": sample["index"],
                "has_image": sample.get("has_image", False),
                "predicted_action": action.tolist(),
                "ground_truth_action": gt_action,
                "inference_time_ms": inference_time,
            }
            
            # Calculate error if ground truth available
            if gt_action is not None:
                gt_array = np.array(gt_action)
                pred_array = np.array(action)
                if gt_array.shape == pred_array.shape:
                    mae = np.mean(np.abs(gt_array - pred_array))
                    result["mae"] = float(mae)
            
            results.append(result)
    
    return results


def print_results(results: list, verbose: bool = True):
    """Pretty print inference results."""
    print("\n" + "=" * 60)
    print("INFERENCE RESULTS")
    print("=" * 60)

    for r in results:
        if "sample_index" in r:
            img_status = "with image" if r.get("has_image") else "no image (synthetic)"
            print(f"\nSample {r['sample_index']} ({img_status}):")
        else:
            print(f"\nStep {r['step']}:")
        
        print(f"  Inference time: {r['inference_time_ms']:.2f}ms")

        action = r.get("predicted_action", r.get("action"))
        print(f"  Predicted action ({len(action)} dims):")

        if verbose and len(action) == 14:
            print(f"    Left arm:      [{', '.join(f'{a:+.4f}' for a in action[0:6])}]")
            print(f"    Left gripper:  {action[6]:+.4f}")
            print(f"    Right arm:     [{', '.join(f'{a:+.4f}' for a in action[7:13])}]")
            print(f"    Right gripper: {action[13]:+.4f}")
        elif verbose and len(action) == 6:
            print(f"    Joints:  [{', '.join(f'{a:+.4f}' for a in action[0:5])}]")
            print(f"    Gripper: {action[5]:+.4f}")
        else:
            if len(action) <= 10:
                print(f"    [{', '.join(f'{a:+.4f}' for a in action)}]")
            else:
                print(f"    [{', '.join(f'{a:+.4f}' for a in action[:6])} ... {', '.join(f'{a:+.4f}' for a in action[-2:])}]")

        # Print ground truth if available
        if "ground_truth_action" in r and r["ground_truth_action"] is not None:
            gt = r["ground_truth_action"]
            print(f"  Ground truth ({len(gt)} dims):")
            if verbose and len(gt) == 14:
                print(f"    Left arm:      [{', '.join(f'{a:+.4f}' for a in gt[0:6])}]")
                print(f"    Left gripper:  {gt[6]:+.4f}")
                print(f"    Right arm:     [{', '.join(f'{a:+.4f}' for a in gt[7:13])}]")
                print(f"    Right gripper: {gt[13]:+.4f}")
            else:
                if len(gt) <= 10:
                    print(f"    [{', '.join(f'{a:+.4f}' for a in gt)}]")
                else:
                    print(f"    [{', '.join(f'{a:+.4f}' for a in gt[:6])} ... {', '.join(f'{a:+.4f}' for a in gt[-2:])}]")
        
        if "mae" in r:
            print(f"  MAE (Mean Absolute Error): {r['mae']:.4f}")

        if "action_chunk_shape" in r:
            print(f"  Action chunk shape: {r['action_chunk_shape']}")

    # Summary
    if len(results) > 0:
        avg_time = sum(r["inference_time_ms"] for r in results) / len(results)
        print(f"\n{'=' * 60}")
        print(f"Average inference time: {avg_time:.2f}ms ({1000/avg_time:.1f} Hz)")
        
        if any("mae" in r for r in results):
            avg_mae = np.mean([r["mae"] for r in results if "mae" in r])
            print(f"Average MAE: {avg_mae:.4f}")
        
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test VLA Model Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--policy-type",
        type=str,
        default="act",
        choices=["act", "smolvla", "diffusion"],
        help="Policy type"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="HuggingFace dataset name (e.g., lerobot/aloha_sim_transfer_cube_human)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to test from dataset"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to test image (uses synthetic if not provided)"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=5,
        help="Number of inference steps to run (when not using dataset)"
    )
    parser.add_argument(
        "--save-images",
        type=str,
        default=None,
        help="Directory to save sample images from dataset"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed action breakdown"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("VLA Model Inference Test - VESSL AI")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Policy type: {args.policy_type}")
    print(f"Device: {args.device}")
    if args.dataset:
        print(f"Dataset: {args.dataset}")
        print(f"Num samples: {args.num_samples}")
    else:
        print(f"Num steps: {args.num_steps}")
    print("=" * 60)

    # Load policy
    print("\nLoading policy...")
    if args.policy_type == "act":
        policy = load_act_policy(args.checkpoint, args.device)
    elif args.policy_type == "smolvla":
        policy = load_smolvla_policy(args.checkpoint, args.device)
    else:
        raise ValueError(f"Unsupported policy type: {args.policy_type}")

    print("Policy loaded successfully!")

    # Run inference
    if args.dataset:
        # Load samples from dataset
        print(f"\nLoading samples from dataset...")
        samples = load_dataset_with_videos(
            args.dataset, 
            args.num_samples,
            args.save_images
        )
        
        if len(samples) > 0:
            print(f"\nRunning inference on {len(samples)} samples...")
            results = run_inference_on_samples(policy, samples, args.device)
        else:
            print("No samples loaded!")
            results = []
    else:
        # Use image or synthetic data
        print("\nPreparing observation...")
        if args.image:
            print(f"Using image: {args.image}")
            obs = load_image_as_observation(args.image, device=args.device)
        else:
            print("Using synthetic observation (random image + zero state)")
            obs = create_dummy_observation(device=args.device)

        print(f"\nRunning {args.num_steps} inference steps...")
        results = run_inference(policy, obs, args.num_steps)

    # Print results
    print_results(results, verbose=args.verbose)

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
