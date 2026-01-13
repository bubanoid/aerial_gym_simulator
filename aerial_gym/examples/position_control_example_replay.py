"""
Replay recorded trajectory with viewer and save as video.
Run with --headless=False on a machine with display.

Usage:
    python position_control_example_replay.py --headless=False --trajectory_file=trajectory_data.pkl
"""
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch
import pickle
import os
from PIL import Image
import numpy as np
from aerial_gym.utils.helpers import get_args, parse_arguments

def get_replay_args():
    custom_parameters = [
        {
            "name": "--trajectory_file",
            "type": str,
            "default": "trajectory_data.pkl",
            "help": "Path to the trajectory file to replay",
        },
        {
            "name": "--output_video",
            "type": str,
            "default": "replay_video.mp4",
            "help": "Output video filename",
        },
        {
            "name": "--save_frames",
            "type": lambda x: bool(x.lower() == 'true'),
            "default": True,
            "help": "Save individual frames as images",
        },
    ]
    args = get_args(additional_parameters=custom_parameters)
    return args


if __name__ == "__main__":
    args = get_replay_args()

    # Load trajectory data
    trajectory_file = args.trajectory_file
    if not os.path.exists(trajectory_file):
        logger.error(f"Trajectory file not found: {trajectory_file}")
        logger.error("Please run position_control_example_record.py first with --headless=True")
        exit(1)

    with open(trajectory_file, "rb") as f:
        trajectory_data = pickle.load(f)

    num_envs = trajectory_data["num_envs"]
    frames = trajectory_data["frames"]
    logger.warning(f"Loaded trajectory with {len(frames)} frames and {num_envs} environments")

    # Must run with headless=False for viewer
    if args.headless:
        logger.error("Replay requires --headless=False to render the viewer")
        logger.error("Run with: python position_control_example_replay.py --headless=False")
        exit(1)

    logger.warning("Building environment with viewer enabled...")
    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="empty_env",
        robot_name="base_quadrotor",
        controller_name="lee_position_control",
        args=None,
        device="cuda:0",
        num_envs=num_envs,
        headless=False,  # Must be False for viewer
        use_warp=args.use_warp,
    )

    # Get references to state tensors and gym/sim objects
    robot_state_tensor = env_manager.global_tensor_dict["robot_state_tensor"]
    gym = env_manager.IGE_env.gym
    sim = env_manager.IGE_env.sim
    viewer = env_manager.IGE_env.viewer.viewer

    # Create frames directory if saving frames
    frames_dir = "replay_frames"
    if args.save_frames:
        os.makedirs(frames_dir, exist_ok=True)
        logger.info(f"Saving frames to {frames_dir}/")

    env_manager.reset()
    saved_frames = []

    logger.warning("Starting replay...")
    for i, frame_data in enumerate(frames):
        # Load recorded robot state
        recorded_state = torch.tensor(frame_data["robot_state"], device="cuda:0")

        # Set the robot state tensor
        robot_state_tensor[:] = recorded_state

        # Write state to simulation
        env_manager.IGE_env.write_to_sim()

        # Step physics to update visuals (with zero actions since we're setting state directly)
        actions = torch.zeros((num_envs, 4)).to("cuda:0")
        env_manager.step(actions=actions)

        # Render viewer
        env_manager.render()

        # Save viewer frame to file
        if args.save_frames:
            frame_path = os.path.join(frames_dir, f"frame_{i:05d}.png")
            gym.write_viewer_image_to_file(viewer, frame_path)

        if (i + 1) % 100 == 0:
            logger.info(f"Replayed {i + 1}/{len(frames)} frames")

    logger.warning(f"Replay complete! {len(frames)} frames rendered.")

    if args.save_frames:
        logger.warning(f"Frames saved to {frames_dir}/")
        logger.warning("To create video, run:")
        logger.warning(f"  ffmpeg -framerate 30 -i {frames_dir}/frame_%05d.png -c:v libx264 -pix_fmt yuv420p {args.output_video}")

    logger.warning("Press ESC in the viewer window to exit, or close the window.")

    # Keep viewer open
    while True:
        env_manager.render()
