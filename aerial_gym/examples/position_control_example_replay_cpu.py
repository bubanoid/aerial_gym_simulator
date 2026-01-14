"""
CPU-based replay of recorded trajectory - no GPU required.
Visualizes quadrotor trajectories using matplotlib.

Usage:
    python position_control_example_replay_cpu.py --trajectory_file=trajectory_data.pkl
"""
import pickle
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation


def quaternion_to_rotation_matrix(q):
    """Convert quaternion [x, y, z, w] to rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])


def create_quadrotor_shape(scale=0.3):
    """Create vertices for a simple quadrotor shape (X configuration)."""
    arm_length = scale
    arm_width = scale * 0.1
    body_size = scale * 0.3

    # Simple X-shaped quadrotor with 4 arms
    vertices = []

    # Center body (cube)
    b = body_size / 2
    body = np.array([
        [-b, -b, -b/2], [b, -b, -b/2], [b, b, -b/2], [-b, b, -b/2],
        [-b, -b, b/2], [b, -b, b/2], [b, b, b/2], [-b, b, b/2]
    ])

    return body


def draw_quadrotor(ax, position, orientation, color='blue', scale=0.3):
    """Draw a quadrotor at given position and orientation."""
    # Get rotation matrix from quaternion
    R = quaternion_to_rotation_matrix(orientation)

    # Arm directions (X configuration)
    arm_length = scale
    arm_dirs = np.array([
        [1, 1, 0],
        [1, -1, 0],
        [-1, -1, 0],
        [-1, 1, 0]
    ]) / np.sqrt(2)

    # Draw arms
    for arm_dir in arm_dirs:
        rotated_dir = R @ arm_dir
        end_point = position + arm_length * rotated_dir
        ax.plot3D(
            [position[0], end_point[0]],
            [position[1], end_point[1]],
            [position[2], end_point[2]],
            color=color, linewidth=2
        )
        # Draw motor circles at arm ends
        ax.scatter3D([end_point[0]], [end_point[1]], [end_point[2]],
                    color=color, s=30, marker='o')

    # Draw body center
    ax.scatter3D([position[0]], [position[1]], [position[2]],
                color=color, s=50, marker='s')

    # Draw orientation arrow (z-axis / up direction)
    z_axis = R @ np.array([0, 0, scale * 0.8])
    ax.quiver(position[0], position[1], position[2],
              z_axis[0], z_axis[1], z_axis[2],
              color='red', arrow_length_ratio=0.3, linewidth=1.5)


def get_args():
    parser = argparse.ArgumentParser(description="CPU-based trajectory replay")
    parser.add_argument("--trajectory_file", type=str, default="trajectory_data.pkl",
                       help="Path to trajectory file")
    parser.add_argument("--output_video", type=str, default="replay_cpu.mp4",
                       help="Output video filename")
    parser.add_argument("--output_gif", type=str, default="replay_cpu.gif",
                       help="Output GIF filename")
    parser.add_argument("--max_envs", type=int, default=16,
                       help="Maximum number of environments to display")
    parser.add_argument("--frame_skip", type=int, default=1,
                       help="Skip frames for faster rendering (1=all frames)")
    parser.add_argument("--fps", type=int, default=30,
                       help="Frames per second for output video")
    parser.add_argument("--save_frames", action="store_true",
                       help="Save individual frames as PNG")
    parser.add_argument("--show_trails", action="store_true",
                       help="Show trajectory trails")
    parser.add_argument("--trail_length", type=int, default=50,
                       help="Length of trajectory trails in frames")
    return parser.parse_args()


def main():
    args = get_args()

    # Load trajectory data
    if not os.path.exists(args.trajectory_file):
        print(f"Error: Trajectory file not found: {args.trajectory_file}")
        print("Please run position_control_example_record.py first")
        return

    print(f"Loading trajectory from {args.trajectory_file}...")
    with open(args.trajectory_file, "rb") as f:
        trajectory_data = pickle.load(f)

    num_envs = min(trajectory_data["num_envs"], args.max_envs)
    frames = trajectory_data["frames"]
    total_frames = len(frames)

    print(f"Loaded {total_frames} frames with {trajectory_data['num_envs']} environments")
    print(f"Displaying {num_envs} environments")

    # Calculate bounds for the plot
    all_positions = []
    for frame in frames:
        positions = frame["robot_state"][:num_envs, :3]
        all_positions.append(positions)
    all_positions = np.array(all_positions)

    min_bounds = all_positions.min(axis=(0, 1)) - 1
    max_bounds = all_positions.max(axis=(0, 1)) + 1

    # Create color map for environments
    colors = plt.cm.tab20(np.linspace(0, 1, num_envs))

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create frames directory if saving frames
    if args.save_frames:
        frames_dir = "replay_frames_cpu"
        os.makedirs(frames_dir, exist_ok=True)
        print(f"Saving frames to {frames_dir}/")

    # Store trail data
    trails = {i: [] for i in range(num_envs)}

    def update(frame_idx):
        ax.clear()

        # Set axis limits
        ax.set_xlim([min_bounds[0], max_bounds[0]])
        ax.set_ylim([min_bounds[1], max_bounds[1]])
        ax.set_zlim([min_bounds[2], max_bounds[2]])

        # Labels
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Quadrotor Trajectory Replay - Frame {frame_idx * args.frame_skip}/{total_frames}')

        # Get frame data
        actual_frame_idx = frame_idx * args.frame_skip
        if actual_frame_idx >= total_frames:
            actual_frame_idx = total_frames - 1

        frame_data = frames[actual_frame_idx]
        robot_states = frame_data["robot_state"]

        # Draw each quadrotor
        for env_id in range(num_envs):
            position = robot_states[env_id, :3]
            orientation = robot_states[env_id, 3:7]  # quaternion [x, y, z, w]

            # Update trail
            trails[env_id].append(position.copy())
            if len(trails[env_id]) > args.trail_length:
                trails[env_id].pop(0)

            # Draw trail
            if args.show_trails and len(trails[env_id]) > 1:
                trail = np.array(trails[env_id])
                ax.plot3D(trail[:, 0], trail[:, 1], trail[:, 2],
                         color=colors[env_id], alpha=0.5, linewidth=1)

            # Draw quadrotor
            draw_quadrotor(ax, position, orientation, color=colors[env_id])

        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])

        return []

    # Calculate number of frames to render
    render_frames = (total_frames + args.frame_skip - 1) // args.frame_skip

    print(f"Rendering {render_frames} frames (frame_skip={args.frame_skip})...")

    if args.save_frames:
        # Save individual frames
        for i in range(render_frames):
            update(i)
            frame_path = os.path.join(frames_dir, f"frame_{i:05d}.png")
            plt.savefig(frame_path, dpi=100, bbox_inches='tight')
            if (i + 1) % 50 == 0:
                print(f"Saved {i + 1}/{render_frames} frames")
        print(f"Frames saved to {frames_dir}/")
        print(f"To create video: ffmpeg -framerate {args.fps} -i {frames_dir}/frame_%05d.png -c:v libx264 -pix_fmt yuv420p {args.output_video}")
    else:
        # Create animation
        print("Creating animation...")
        anim = animation.FuncAnimation(
            fig, update, frames=render_frames,
            interval=1000/args.fps, blit=False
        )

        # Save video
        print(f"Saving video to {args.output_video}...")
        try:
            writer = animation.FFMpegWriter(fps=args.fps, bitrate=2000)
            anim.save(args.output_video, writer=writer)
            print(f"Video saved: {args.output_video}")
        except Exception as e:
            print(f"Could not save MP4 (ffmpeg may not be installed): {e}")
            print(f"Trying to save as GIF instead...")
            try:
                anim.save(args.output_gif, writer='pillow', fps=args.fps)
                print(f"GIF saved: {args.output_gif}")
            except Exception as e2:
                print(f"Could not save GIF: {e2}")
                print("Falling back to saving individual frames...")
                args.save_frames = True
                frames_dir = "replay_frames_cpu"
                os.makedirs(frames_dir, exist_ok=True)
                for i in range(render_frames):
                    update(i)
                    plt.savefig(os.path.join(frames_dir, f"frame_{i:05d}.png"), dpi=100)
                print(f"Frames saved to {frames_dir}/")

    print("Done!")

    # Show interactive plot at the end
    print("Showing final frame (close window to exit)...")
    update(render_frames - 1)
    plt.show()


if __name__ == "__main__":
    main()
