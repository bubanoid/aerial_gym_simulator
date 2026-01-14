"""
Position control example with trajectory recording.
Run with --headless=True on remote machine to record simulation data.
The recorded trajectory can be replayed using position_control_example_replay.py
"""
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch
import pickle
from aerial_gym.utils.helpers import get_args

if __name__ == "__main__":
    args = get_args()
    logger.warning("Recording position control example trajectory.")
    logger.warning(f"Headless mode: {args.headless}")

    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="empty_env",
        robot_name="base_quadrotor",
        controller_name="lee_position_control",
        args=None,
        device="cuda:0",
        num_envs=args.num_envs,
        headless=args.headless,
        use_warp=args.use_warp,
    )

    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")
    env_manager.reset()

    # Store trajectory data
    trajectory_data = {
        "num_envs": env_manager.num_envs,
        "frames": []
    }

    num_steps = 1000  # Adjust as needed

    for i in range(num_steps):
        if i % 1000 == 0:
            logger.info(f"Step {i}, changing target setpoint.")
            env_manager.reset()

        env_manager.step(actions=actions)

        # Record robot state for all environments
        # robot_state_tensor format: [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, vel_x, vel_y, vel_z, angvel_x, angvel_y, angvel_z]
        frame_data = {
            "robot_state": env_manager.global_tensor_dict["robot_state_tensor"].cpu().numpy().copy(),
        }
        trajectory_data["frames"].append(frame_data)

        if (i + 1) % 100 == 0:
            logger.info(f"Recorded {i + 1}/{num_steps} frames")

    # Save trajectory to file
    output_file = "trajectory_data.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(trajectory_data, f)

    logger.warning(f"Trajectory saved to {output_file}")
    logger.warning(f"Total frames: {len(trajectory_data['frames'])}")
    logger.warning(f"Number of environments: {trajectory_data['num_envs']}")
    logger.warning("To replay, run: python position_control_example_replay.py --headless=False")
