#!/usr/bin/env python3
"""
ALOHA Simulation Server with ACT Policy
VESSL AI - Physical AI Hackathon Demo

Test ACT policies trained on lerobot/aloha_sim_transfer_cube_human
in gym-aloha simulation environment.
"""

# Set up headless rendering BEFORE importing mujoco/gymnasium
import os
os.environ.setdefault("MUJOCO_GL", "egl")  # Use EGL for headless rendering

import argparse
import asyncio
import base64
import io
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# Web server
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

# PyTorch
import torch

# Try to import gymnasium
try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    print("Warning: gymnasium not installed. Run: pip install gymnasium")

# Try to import gym-aloha
try:
    import gym_aloha
    HAS_GYM_ALOHA = True
except ImportError:
    HAS_GYM_ALOHA = False
    print("Warning: gym-aloha not installed. Run: pip install gym-aloha")

# Try to import LeRobot
try:
    from lerobot.common.policies.act.modeling_act import ACTPolicy
    from lerobot.common.policies.act.configuration_act import ACTConfig
    HAS_LEROBOT = True
except ImportError:
    HAS_LEROBOT = False
    print("Warning: lerobot not installed")


app = FastAPI(title="ALOHA Simulation - VESSL AI")

# Global state
SIM_STATE = {
    "env": None,
    "env_name": "AlohaTransferCube-v0",
    "policy": None,
    "policy_name": None,
    "running": False,
    "mode": "random",  # "random" or "policy"
    "frame_count": 0,
    "episode_count": 0,
    "success_count": 0,
    "last_reward": 0.0,
    "last_success": False,
    "connections": [],
}


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ALOHA Simulation - VESSL AI</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            background: #0f0f1a;
            color: #eee;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            color: #00d4ff;
            margin-bottom: 5px;
            font-size: 24px;
        }
        .subtitle {
            color: #888;
            margin-bottom: 20px;
            font-size: 14px;
        }
        .main-layout {
            display: flex;
            gap: 20px;
        }
        .video-section {
            flex: 2;
        }
        #simulation-frame {
            width: 100%;
            max-width: 640px;
            border: 2px solid #333;
            border-radius: 8px;
            background: #1a1a2e;
        }
        .info-bar {
            display: flex;
            gap: 15px;
            margin-top: 10px;
            font-size: 13px;
            color: #888;
        }
        .info-item {
            background: #1a1a2e;
            padding: 5px 12px;
            border-radius: 4px;
        }
        .info-item.success {
            background: #1e5128;
            color: #4ade80;
        }
        .info-item.fail {
            background: #5c1a1a;
            color: #f87171;
        }
        .controls {
            flex: 1;
            min-width: 280px;
        }
        .panel {
            background: #1a1a2e;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        .panel h3 {
            margin: 0 0 12px 0;
            font-size: 14px;
            color: #00d4ff;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .status {
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
            margin-bottom: 12px;
        }
        .status.connected { background: #1e5128; color: #4ade80; }
        .status.disconnected { background: #5c1a1a; color: #f87171; }

        select, button {
            width: 100%;
            padding: 10px 15px;
            border: none;
            border-radius: 5px;
            font-size: 14px;
            cursor: pointer;
            margin-bottom: 8px;
        }
        select {
            background: #252540;
            color: #eee;
        }
        button {
            background: #00d4ff;
            color: #000;
            font-weight: 600;
        }
        button:hover { background: #00a8cc; }
        button:disabled {
            background: #444;
            color: #888;
            cursor: not-allowed;
        }
        button.secondary {
            background: #333;
            color: #eee;
        }
        button.secondary:hover { background: #444; }

        .stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }
        .stat-item {
            background: #252540;
            padding: 10px;
            border-radius: 5px;
        }
        .stat-label {
            font-size: 11px;
            color: #888;
            text-transform: uppercase;
        }
        .stat-value {
            font-size: 18px;
            font-weight: bold;
            color: #fff;
        }
        .stat-value.highlight { color: #00d4ff; }

        .policy-status {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            background: #252540;
            border-radius: 5px;
            margin-bottom: 12px;
        }
        .policy-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #f87171;
        }
        .policy-dot.loaded { background: #4ade80; }
        .policy-name {
            font-size: 12px;
            color: #888;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .episode-result {
            padding: 12px;
            border-radius: 5px;
            margin-top: 10px;
            display: none;
        }
        .episode-result.show { display: block; }
        .episode-result.success { background: #1e5128; border: 1px solid #4ade80; }
        .episode-result.fail { background: #3d1a1a; border: 1px solid #f87171; }
    </style>
</head>
<body>
    <div class="container">
        <h1>ALOHA Transfer Cube Simulation</h1>
        <p class="subtitle">ACT Policy Testing | Powered by VESSL AI</p>

        <div class="main-layout">
            <div class="video-section">
                <img id="simulation-frame" src="" alt="Simulation">
                <div class="info-bar">
                    <span class="info-item">Step: <span id="step-count">0</span></span>
                    <span class="info-item">FPS: <span id="fps">0</span></span>
                    <span class="info-item" id="last-result">-</span>
                </div>
            </div>

            <div class="controls">
                <div class="panel">
                    <div id="status" class="status disconnected">Disconnected</div>

                    <h3>Action Mode</h3>
                    <select id="mode-select" onchange="setMode(this.value)">
                        <option value="random">Random Actions</option>
                        <option value="policy">ACT Policy</option>
                    </select>

                    <div class="policy-status">
                        <div id="policy-dot" class="policy-dot"></div>
                        <span id="policy-name" class="policy-name">No policy loaded</span>
                    </div>

                    <button onclick="runEpisode()">Run Episode</button>
                    <button class="secondary" onclick="toggleSimulation()" id="toggle-btn">Start</button>
                    <button class="secondary" onclick="resetEnvironment()">Reset</button>
                </div>

                <div class="panel">
                    <h3>Statistics</h3>
                    <div class="stat-grid">
                        <div class="stat-item">
                            <div class="stat-label">Episodes</div>
                            <div class="stat-value" id="episode-count">0</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Success Rate</div>
                            <div class="stat-value highlight" id="success-rate">0%</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Last Reward</div>
                            <div class="stat-value" id="last-reward">-</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Frames</div>
                            <div class="stat-value" id="frame-count">0</div>
                        </div>
                    </div>
                </div>

                <div id="episode-result" class="episode-result">
                    <strong id="result-title">Episode Complete</strong>
                    <div id="result-details"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let isRunning = false;
        let lastFrameTime = Date.now();

        function connect() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);

            ws.onopen = () => {
                document.getElementById('status').className = 'status connected';
                document.getElementById('status').textContent = 'Connected';
            };

            ws.onclose = () => {
                document.getElementById('status').className = 'status disconnected';
                document.getElementById('status').textContent = 'Disconnected';
                setTimeout(connect, 2000);
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);

                if (data.type === 'frame') {
                    document.getElementById('simulation-frame').src = 'data:image/jpeg;base64,' + data.image;
                    const now = Date.now();
                    const fps = Math.round(1000 / (now - lastFrameTime));
                    lastFrameTime = now;
                    document.getElementById('fps').textContent = fps;
                    if (data.step !== undefined) {
                        document.getElementById('step-count').textContent = data.step;
                    }
                }

                if (data.type === 'stats') {
                    document.getElementById('episode-count').textContent = data.episode_count || 0;
                    document.getElementById('frame-count').textContent = data.frame_count || 0;
                    document.getElementById('last-reward').textContent =
                        data.last_reward !== undefined ? data.last_reward.toFixed(2) : '-';

                    if (data.episode_count > 0) {
                        const rate = ((data.success_count / data.episode_count) * 100).toFixed(1);
                        document.getElementById('success-rate').textContent = rate + '%';
                    }

                    // Update policy status
                    const policyDot = document.getElementById('policy-dot');
                    const policyName = document.getElementById('policy-name');
                    if (data.policy_loaded) {
                        policyDot.classList.add('loaded');
                        policyName.textContent = data.policy_name || 'Loaded';
                    } else {
                        policyDot.classList.remove('loaded');
                        policyName.textContent = 'No policy loaded';
                    }

                    // Update mode selector
                    document.getElementById('mode-select').value = data.mode || 'random';
                }

                if (data.type === 'episode_complete') {
                    showEpisodeResult(data);
                }
            };
        }

        function showEpisodeResult(data) {
            const resultDiv = document.getElementById('episode-result');
            const title = document.getElementById('result-title');
            const details = document.getElementById('result-details');

            resultDiv.className = 'episode-result show ' + (data.success ? 'success' : 'fail');
            title.textContent = data.success ? 'Success!' : 'Episode Complete';
            details.innerHTML = `
                Steps: ${data.steps} | Reward: ${data.total_reward.toFixed(2)} | Mode: ${data.mode}
            `;

            // Update last result indicator
            const lastResult = document.getElementById('last-result');
            lastResult.textContent = data.success ? 'Last: SUCCESS' : 'Last: -';
            lastResult.className = 'info-item ' + (data.success ? 'success' : '');

            setTimeout(() => {
                resultDiv.classList.remove('show');
            }, 5000);
        }

        function setMode(mode) {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({action: 'set_mode', mode: mode}));
            }
        }

        function toggleSimulation() {
            isRunning = !isRunning;
            const btn = document.getElementById('toggle-btn');
            btn.textContent = isRunning ? 'Stop' : 'Start';
            if (ws) ws.send(JSON.stringify({action: isRunning ? 'start' : 'stop'}));
        }

        function resetEnvironment() {
            if (ws) ws.send(JSON.stringify({action: 'reset'}));
        }

        function runEpisode() {
            if (ws) ws.send(JSON.stringify({action: 'run_episode'}));
        }

        connect();
    </script>
</body>
</html>
"""


class ACTPolicyWrapper:
    """Wrapper for loading and running ACT policy."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """
        Load ACT policy from checkpoint or HuggingFace.

        Args:
            checkpoint_path: Local path or HuggingFace model ID
            device: Device for inference
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.policy = None
        self.name = checkpoint_path
        self._load_policy(checkpoint_path)

    def _load_policy(self, checkpoint_path: str):
        """Load policy from checkpoint."""
        if not HAS_LEROBOT:
            raise RuntimeError("lerobot not installed")

        path = Path(checkpoint_path)

        # Try loading from HuggingFace pretrained
        if not path.exists():
            print(f"Loading from HuggingFace: {checkpoint_path}")
            try:
                self.policy = ACTPolicy.from_pretrained(checkpoint_path)
                self.policy.to(self.device)
                self.policy.eval()
                print(f"Loaded policy from HuggingFace: {checkpoint_path}")
                return
            except Exception as e:
                print(f"Failed to load from HuggingFace: {e}")
                raise

        # Load from local checkpoint
        print(f"Loading from local checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Get config from checkpoint
        config_dict = checkpoint.get("config", {})
        model_config = config_dict.get("model", {})

        # Create ACT config
        act_config = ACTConfig(
            chunk_size=model_config.get("chunk_size", 100),
            n_obs_steps=model_config.get("n_obs_steps", 1),
        )

        # Create and load model
        self.policy = ACTPolicy(act_config)
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.policy.to(self.device)
        self.policy.eval()
        print(f"Loaded policy from checkpoint: {checkpoint_path}")

    def reset(self):
        """Reset policy state (action queue)."""
        if self.policy is not None:
            self.policy.reset()

    def select_action(self, observation: dict) -> np.ndarray:
        """
        Select action given gym-aloha observation.

        Args:
            observation: Dict with 'pixels' and 'agent_pos'

        Returns:
            14-dim action array
        """
        if self.policy is None:
            raise RuntimeError("Policy not loaded")

        # Preprocess observation
        batch = self._preprocess_observation(observation)

        # Run inference
        with torch.no_grad():
            action = self.policy.select_action(batch)

        return action.cpu().numpy().squeeze()

    def _preprocess_observation(self, obs: dict) -> dict:
        """Convert gym-aloha observation to policy input format."""
        import einops

        batch = {}

        # Process images
        if "pixels" in obs:
            pixels = obs["pixels"]
            if isinstance(pixels, dict):
                # Multi-camera: use 'top' camera
                for cam_name, img in pixels.items():
                    img_tensor = torch.from_numpy(img).float()
                    img_tensor = einops.rearrange(img_tensor, "h w c -> 1 c h w")
                    img_tensor = img_tensor / 255.0
                    batch[f"observation.images.{cam_name}"] = img_tensor.to(self.device)
            else:
                img_tensor = torch.from_numpy(pixels).float()
                img_tensor = einops.rearrange(img_tensor, "h w c -> 1 c h w")
                img_tensor = img_tensor / 255.0
                batch["observation.images.top"] = img_tensor.to(self.device)

        # Process robot state (joint positions)
        if "agent_pos" in obs:
            state = torch.from_numpy(obs["agent_pos"]).float()
            state = state.unsqueeze(0)
            batch["observation.state"] = state.to(self.device)

        return batch


class AlohaSimulationEnvironment:
    """Wrapper for gym-aloha AlohaTransferCube-v0 environment."""

    def __init__(self, obs_type: str = "pixels_agent_pos"):
        """
        Create ALOHA simulation environment.

        Args:
            obs_type: "pixels", "pixels_agent_pos", or "state"
        """
        self.obs_type = obs_type
        self.env = None
        self.obs = None
        self.episode_reward = 0.0
        self.step_count = 0
        self.success = False

    def create(self) -> bool:
        """Create the environment."""
        if not HAS_GYM:
            print("gymnasium not installed")
            return False

        if not HAS_GYM_ALOHA:
            print("gym-aloha not installed, using Pendulum fallback")
            self.env = gym.make("Pendulum-v1", render_mode="rgb_array")
            self.obs, _ = self.env.reset()
            return True

        try:
            self.env = gym.make(
                "gym_aloha/AlohaTransferCube-v0",
                obs_type=self.obs_type,
                render_mode="rgb_array",
            )
            self.obs, _ = self.env.reset()
            print(f"Created AlohaTransferCube-v0 environment (obs_type={self.obs_type})")
            return True
        except Exception as e:
            print(f"Failed to create ALOHA environment: {e}")
            print("Falling back to Pendulum-v1")
            self.env = gym.make("Pendulum-v1", render_mode="rgb_array")
            self.obs, _ = self.env.reset()
            return True

    def step(self, action=None):
        """Take a step in the environment."""
        if self.env is None:
            return None, 0, False, {}

        if action is None:
            action = self.env.action_space.sample()

        # Ensure action is numpy array with correct shape
        action = np.asarray(action, dtype=np.float32)

        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += reward
        self.step_count += 1
        self.success = info.get("is_success", False)
        done = terminated or truncated

        return self.obs, reward, done, info

    def reset(self):
        """Reset the environment."""
        if self.env is None:
            return None

        self.obs, _ = self.env.reset()
        self.episode_reward = 0.0
        self.step_count = 0
        self.success = False
        return self.obs

    def render(self) -> np.ndarray:
        """Render the environment."""
        if self.env is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        try:
            frame = self.env.render()
            if frame is None:
                return np.zeros((480, 640, 3), dtype=np.uint8)
            return frame
        except Exception:
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def close(self):
        """Close the environment."""
        if self.env is not None:
            self.env.close()


def frame_to_base64(frame: np.ndarray, quality: int = 70) -> str:
    """Convert numpy frame to base64 JPEG."""
    img = Image.fromarray(frame)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode()


@app.get("/")
async def root():
    return HTMLResponse(HTML_TEMPLATE)


@app.get("/api/status")
async def api_status():
    """Get current simulation status."""
    return {
        "env_name": SIM_STATE["env_name"],
        "running": SIM_STATE["running"],
        "mode": SIM_STATE["mode"],
        "frame_count": SIM_STATE["frame_count"],
        "episode_count": SIM_STATE["episode_count"],
        "success_count": SIM_STATE["success_count"],
        "policy_loaded": SIM_STATE["policy"] is not None,
        "policy_name": SIM_STATE["policy_name"],
        "connections": len(SIM_STATE["connections"]),
    }


@app.post("/api/load_policy")
async def api_load_policy(checkpoint: str, device: str = "cuda"):
    """Load an ACT policy checkpoint."""
    try:
        policy = ACTPolicyWrapper(checkpoint, device=device)
        SIM_STATE["policy"] = policy
        SIM_STATE["policy_name"] = checkpoint
        return {"status": "loaded", "checkpoint": checkpoint}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/run_episode")
async def api_run_episode():
    """Run a complete episode and return results."""
    if SIM_STATE["env"] is None:
        return {"error": "No environment loaded"}

    env = SIM_STATE["env"]
    policy = SIM_STATE["policy"]
    mode = SIM_STATE["mode"]

    env.reset()
    if policy:
        policy.reset()

    done = False
    max_steps = 400

    while not done and env.step_count < max_steps:
        if mode == "policy" and policy is not None:
            try:
                action = policy.select_action(env.obs)
            except Exception:
                action = None
        else:
            action = None

        _, _, done, _ = env.step(action)

    SIM_STATE["episode_count"] += 1
    if env.success:
        SIM_STATE["success_count"] += 1
    SIM_STATE["last_reward"] = env.episode_reward
    SIM_STATE["last_success"] = env.success

    return {
        "steps": env.step_count,
        "total_reward": env.episode_reward,
        "success": env.success,
        "episode": SIM_STATE["episode_count"],
        "mode": mode,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    SIM_STATE["connections"].append(websocket)

    try:
        # Send initial stats
        await send_stats(websocket)

        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=0.05)
                await handle_command(websocket, data)
            except asyncio.TimeoutError:
                pass

            # Stream frames if running
            if SIM_STATE["running"] and SIM_STATE["env"]:
                await stream_frame(websocket)

            await asyncio.sleep(1/30)  # 30 FPS

    except WebSocketDisconnect:
        if websocket in SIM_STATE["connections"]:
            SIM_STATE["connections"].remove(websocket)


async def handle_command(websocket: WebSocket, data: dict):
    """Handle WebSocket commands."""
    action = data.get("action")

    if action == "start":
        SIM_STATE["running"] = True

    elif action == "stop":
        SIM_STATE["running"] = False

    elif action == "reset":
        if SIM_STATE["env"]:
            SIM_STATE["env"].reset()
            if SIM_STATE["policy"]:
                SIM_STATE["policy"].reset()
        await send_stats(websocket)

    elif action == "set_mode":
        SIM_STATE["mode"] = data.get("mode", "random")
        await send_stats(websocket)

    elif action == "run_episode":
        await run_episode_streaming(websocket)


async def stream_frame(websocket: WebSocket):
    """Stream a single simulation frame."""
    env = SIM_STATE["env"]
    policy = SIM_STATE["policy"]
    mode = SIM_STATE["mode"]

    # Get action
    if mode == "policy" and policy is not None:
        try:
            action = policy.select_action(env.obs)
        except Exception:
            action = None
    else:
        action = None

    # Step
    _, _, done, _ = env.step(action)

    if done:
        SIM_STATE["episode_count"] += 1
        if env.success:
            SIM_STATE["success_count"] += 1
        SIM_STATE["last_reward"] = env.episode_reward
        SIM_STATE["last_success"] = env.success
        env.reset()
        if policy:
            policy.reset()

    # Send frame
    frame = env.render()
    await websocket.send_json({
        "type": "frame",
        "image": frame_to_base64(frame),
        "step": env.step_count,
    })
    SIM_STATE["frame_count"] += 1

    # Send stats periodically
    if SIM_STATE["frame_count"] % 30 == 0:
        await send_stats(websocket)


async def run_episode_streaming(websocket: WebSocket):
    """Run episode with frame streaming."""
    if SIM_STATE["env"] is None:
        return

    env = SIM_STATE["env"]
    policy = SIM_STATE["policy"]
    mode = SIM_STATE["mode"]

    env.reset()
    if policy:
        policy.reset()

    done = False
    max_steps = 400

    while not done and env.step_count < max_steps:
        # Get action
        if mode == "policy" and policy is not None:
            try:
                action = policy.select_action(env.obs)
            except Exception:
                action = None
        else:
            action = None

        # Step
        _, _, done, _ = env.step(action)

        # Stream frame
        frame = env.render()
        await websocket.send_json({
            "type": "frame",
            "image": frame_to_base64(frame),
            "step": env.step_count,
        })
        SIM_STATE["frame_count"] += 1

        await asyncio.sleep(1/30)

    # Update stats
    SIM_STATE["episode_count"] += 1
    if env.success:
        SIM_STATE["success_count"] += 1
    SIM_STATE["last_reward"] = env.episode_reward
    SIM_STATE["last_success"] = env.success

    # Send completion
    await websocket.send_json({
        "type": "episode_complete",
        "steps": env.step_count,
        "total_reward": env.episode_reward,
        "success": env.success,
        "episode": SIM_STATE["episode_count"],
        "mode": mode,
    })

    await send_stats(websocket)


async def send_stats(websocket: WebSocket):
    """Send current statistics."""
    await websocket.send_json({
        "type": "stats",
        "env_name": SIM_STATE["env_name"],
        "frame_count": SIM_STATE["frame_count"],
        "episode_count": SIM_STATE["episode_count"],
        "success_count": SIM_STATE["success_count"],
        "last_reward": SIM_STATE["last_reward"],
        "last_success": SIM_STATE["last_success"],
        "policy_loaded": SIM_STATE["policy"] is not None,
        "policy_name": SIM_STATE["policy_name"],
        "mode": SIM_STATE["mode"],
    })


def main():
    parser = argparse.ArgumentParser(description="ALOHA Simulation Server")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--checkpoint", default=None,
                       help="ACT policy checkpoint (path or HuggingFace model ID)")
    parser.add_argument("--device", default="cuda", help="Device for inference")
    parser.add_argument("--obs-type", default="pixels_agent_pos",
                       help="Observation type: pixels, pixels_agent_pos, state")
    args = parser.parse_args()

    print("=" * 60)
    print("ALOHA Simulation Server - VESSL AI")
    print("=" * 60)

    # Create environment
    print(f"\nCreating environment (obs_type={args.obs_type})...")
    env = AlohaSimulationEnvironment(obs_type=args.obs_type)
    if env.create():
        SIM_STATE["env"] = env
        print("Environment ready!")
    else:
        print("Warning: Could not create environment")
        sys.exit(1)

    # Load policy if specified
    if args.checkpoint:
        print(f"\nLoading ACT policy: {args.checkpoint}")
        try:
            policy = ACTPolicyWrapper(args.checkpoint, device=args.device)
            SIM_STATE["policy"] = policy
            SIM_STATE["policy_name"] = args.checkpoint
            SIM_STATE["mode"] = "policy"
            print("Policy loaded!")
        except Exception as e:
            print(f"Failed to load policy: {e}")
            print("Continuing with random actions only")

    print(f"\nDevice: {args.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    print(f"\nStarting server on http://0.0.0.0:{args.port}")
    print("Open in browser to view simulation")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
