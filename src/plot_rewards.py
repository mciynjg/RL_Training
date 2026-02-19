"""
Plot training reward curves from TensorBoard logs.
"""

import os
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np

try:
    from tensorboard.backend.event_processing import event_accumulator
    HAS_TB = True
except ImportError:
    HAS_TB = False
    print("Warning: tensorboard not installed. Install with: pip install tensorboard")

import matplotlib
import matplotlib.pyplot as plt


def find_log_directories(log_dir: str = "./log") -> List[str]:
    """
    Find all TensorBoard log directories.

    Args:
        log_dir: Base log directory

    Returns:
        List of log directory paths
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        return []

    # If log_dir is explicitly a directory containing events file, return it
    if log_path.is_dir() and list(log_path.glob("events.out.tfevents.*")):
        return [str(log_path)]
    
    # If log_dir is a file, return its parent directory
    if log_path.is_file() and "events.out.tfevents" in log_path.name:
        return [str(log_path.parent)]

    # Find directories with event files
    log_dirs = []
    # Use rglob to find all event files
    for item in log_path.rglob("events.out.tfevents.*"):
        if item.parent not in [Path(p) for p in log_dirs]:
            log_dirs.append(str(item.parent))

    return sorted(log_dirs)


def extract_rewards(log_path: str) -> Dict[str, Dict[str, List[Any]]]:
    """
    Extract rewards from TensorBoard log.

    Args:
        log_path: Path to TensorBoard log directory

    Returns:
        Dictionary with scalar data
    """
    if not HAS_TB:
        return {}

    try:
        ea = event_accumulator.EventAccumulator(log_path)
        ea.Reload()

        data = {}

        # Get all scalar tags
        try:
            scalar_tags = ea.Tags().get("scalars", [])
        except Exception as e:
            print(f"Warning: Could not get tags from {log_path}: {e}")
            return {}
        
        if not scalar_tags:
            # print(f"Warning: No scalar tags found in {log_path}")
            return {}

        for tag in scalar_tags:
            try:
                events = ea.Scalars(tag)
                data[tag] = {
                    "steps": [e.step for e in events],
                    "values": [e.value for e in events],
                    "times": [e.wall_time for e in events],
                }
            except Exception as e:
                # print(f"Error reading tag {tag} in {log_path}: {e}")
                continue

        return data
    except Exception as e:
        print(f"Error loading log {log_path}: {e}")
        return {}



def parse_log_name(log_path: str) -> Dict[str, str]:
    """
    Parse algorithm and environment from log directory name.

    Args:
        log_path: Path to log directory

    Returns:
        Dictionary with algorithm and environment
    """
    name = Path(log_path).name
    parts = name.split("_")

    result = {
        "algorithm": "unknown",
        "environment": "unknown",
        "timestamp": "",
    }

    if len(parts) >= 2:
        result["algorithm"] = parts[0]
        result["environment"] = "_".join(parts[1:-1]) if len(parts) > 2 else parts[1]
        if len(parts) > 2 and parts[-1].isdigit():
            result["timestamp"] = parts[-1]
            
    return result


def plot_rewards(
    log_paths: Optional[List[str]] = None,
    log_dir: str = "./log",
    save_path: Optional[str] = None,
    show: bool = True,
    smooth_window: int = 10,
) -> Optional[plt.Figure]:
    """
    Plot reward curves from TensorBoard logs.

    Args:
        log_paths: List of log directory paths (optional, will auto-discover if None)
        log_dir: Base log directory for auto-discovery
        save_path: Path to save figure (optional)
        show: Whether to display the plot
        smooth_window: Window size for moving average smoothing

    Returns:
        Matplotlib figure or None
    """
    # Auto-discover log directories if not specified
    if log_paths is None:
        log_paths = find_log_directories(log_dir)

    if not log_paths:
        print(f"No log directories found in {log_dir}")
        return None

    # Filter valid logs and group by environment
    env_groups: Dict[str, List] = {"paths": [], "infos": [], "data": []}
    
    # Track if we found any valid data
    found_any_data = False

    for log_path in log_paths:
        data = extract_rewards(log_path)
        if not data:
            continue

        info = parse_log_name(log_path)

        # Check if this log has valid reward data
        has_data = False
        
        # Debug: Print available tags if no standard keys found
        available_tags = list(data.keys())
        
        possible_keys = [
            'training/returns_stat/mean', 'test/returns_stat/mean',
            'train/reward', 'test/reward', 
            'training/returns', 'test/returns',
            'train/rew', 'test/rew',
            'collect/rew', 'eval/rew',
            'return', 'returns', 'reward', 'rew'
        ]
        
        for tag in possible_keys:
            if tag in data and len(data[tag].get('values', [])) > 0:
                has_data = True
                break
        
        # Fallback check: look for partial matches
        if not has_data:
             for key in available_tags:
                key_lower = key.lower()
                if ('rew' in key_lower or 'reward' in key_lower or 'return' in key_lower) and len(data[key].get('values', [])) > 0:
                    has_data = True
                    break

        if has_data:
            env_groups["paths"].append(log_path)
            env_groups["infos"].append(info)
            env_groups["data"].append(data)
            found_any_data = True
        else:
            # Only print verbose warning if we haven't found any data yet, to avoid spam
            # print(f"Skipping {log_path}: No reward data found.")
            pass

    if not env_groups["paths"]:
        print(f"No valid training logs found with reward data.")
        return None

    # Group by environment
    by_env: Dict[str, Dict] = {}
    for i, info in enumerate(env_groups["infos"]):
        env_name = info["environment"]
        if env_name not in by_env:
            by_env[env_name] = {"paths": [], "infos": [], "data": [], "indices": []}
        by_env[env_name]["paths"].append(env_groups["paths"][i])
        by_env[env_name]["infos"].append(info)
        by_env[env_name]["data"].append(env_groups["data"][i])
        by_env[env_name]["indices"].append(i)

    # Determine figure layout
    num_envs = len(by_env)
    if num_envs == 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        env_list = list(by_env.keys())
        axes_by_env = {env_list[0]: axes}
    else:
        fig, axes = plt.subplots(num_envs, 2, figsize=(14, 5 * num_envs))
        # Handle case where subplots returns 1D array (if 1 row) or 2D
        if num_envs == 1:
             axes = np.array([axes])
        
        # Ensure axes is 2D array [rows, cols]
        if len(axes.shape) == 1:
             # This happens if we have multiple envs but only 1 col? No, we requested 2 cols.
             # If num_envs > 1, axes is (num_envs, 2).
             pass
             
        env_list = sorted(by_env.keys())
        axes_by_env = {env: axes[i] for i, env in enumerate(env_list)}

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Plot each environment
    for env_idx, (env_name, env_data) in enumerate(by_env.items()):
        current_axes = axes_by_env[env_name]
        # Handle 1D array of axes (if only 1 env)
        if hasattr(current_axes, 'shape') and len(current_axes.shape) == 1:
            ax1, ax2 = current_axes
        elif isinstance(current_axes, np.ndarray):
             ax1, ax2 = current_axes
        else:
             ax1, ax2 = current_axes

        plotted_train = 0
        plotted_test = 0

        # Plot training rewards
        for idx, (log_path, info, data) in enumerate(zip(env_data["paths"], env_data["infos"], env_data["data"])):
            reward_key = None
            possible_keys = [
                'training/returns_stat/mean', 'training/returns',
                'train/returns_stat/mean', 'train/rew', 'train/reward',
                'training/rew', 'training/reward', 'collect/rew', 'collect/reward',
                'return', 'returns', 'reward', 'rew'
            ]
            for key in possible_keys:
                if key in data and len(data[key].get('values', [])) > 0:
                    reward_key = key
                    break
            
            # Fallback search
            if reward_key is None:
                for key in data.keys():
                    key_lower = key.lower()
                    if ('rew' in key_lower or 'reward' in key_lower or 'return' in key_lower) and \
                       ('train' in key_lower or 'collect' in key_lower or 'training' in key_lower or 'mean' in key_lower):
                        if len(data[key].get('values', [])) > 0:
                            reward_key = key
                            break

            if reward_key and data[reward_key]:
                values = data[reward_key]["values"]
                steps = data[reward_key]["steps"]

                # Apply smoothing (use smaller window if data is scarce)
                actual_window = min(smooth_window, max(1, len(values) // 3))
                if len(values) >= actual_window and actual_window > 1:
                    values_smooth = np.convolve(values, np.ones(actual_window) / actual_window, mode="valid")
                    steps_smooth = steps[actual_window - 1:]
                else:
                    values_smooth = values
                    steps_smooth = steps

                color_idx = idx % 10
                label = f"{info['algorithm']}"
                # Add timestamp to label if multiple runs of same algo
                # Check if there are other runs with same algo
                same_algo_count = sum(1 for i in env_data["infos"] if i['algorithm'] == info['algorithm'])
                if same_algo_count > 1 and info['timestamp']:
                     label += f" ({info['timestamp'][-4:]})"

                ax1.plot(steps_smooth, values_smooth, label=label, color=colors[color_idx], alpha=0.8, linewidth=2)
                plotted_train += 1

        ax1.set_xlabel("Training Steps")
        ax1.set_ylabel("Reward (smoothed)")
        ax1.set_title(f"Training Rewards - {env_name}")
        if plotted_train > 0:
            ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)

        # Plot test rewards
        for idx, (log_path, info, data) in enumerate(zip(env_data["paths"], env_data["infos"], env_data["data"])):
            test_reward_key = None
            possible_keys = [
                'test/returns_stat/mean', 'test/returns',
                'test/rew', 'test/reward',
                'eval/returns_stat/mean', 'eval/rew', 'eval/reward',
            ]

            for key in possible_keys:
                if key in data and len(data[key].get('values', [])) > 0:
                    test_reward_key = key
                    break
            
            # Also try to find keys starting with test/ or eval/
            if test_reward_key is None:
                 for key in data.keys():
                    if (key.startswith('test/') or key.startswith('eval/')) and \
                       ('rew' in key or 'return' in key):
                        if len(data[key].get('values', [])) > 0:
                            test_reward_key = key
                            break

            if test_reward_key and data[test_reward_key]:
                values = data[test_reward_key]["values"]
                steps = data[test_reward_key]["steps"]

                color_idx = idx % 10
                label = f"{info['algorithm']}"
                # Add timestamp to label if multiple runs of same algo
                same_algo_count = sum(1 for i in env_data["infos"] if i['algorithm'] == info['algorithm'])
                if same_algo_count > 1 and info['timestamp']:
                     label += f" ({info['timestamp'][-4:]})"
                     
                ax2.plot(steps, values, "o-", label=label, color=colors[color_idx], alpha=0.8, markersize=4, linewidth=2)
                plotted_test += 1

        ax2.set_xlabel("Epoch/Step")
        ax2.set_ylabel("Test Reward")
        ax2.set_title(f"Test Rewards - {env_name}")
        if plotted_test > 0:
            ax2.legend(loc="best")
        ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Training Results - {len(env_groups['paths'])} runs ({num_envs} environments)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()

    return fig


def compare_runs(
    log_dir: str = "./log",
    algorithms: Optional[List[str]] = None,
    environment: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Compare specific algorithm runs.

    Args:
        log_dir: Base log directory
        algorithms: List of algorithms to compare
        environment: Specific environment to filter
        save_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure or None
    """
    all_logs = find_log_directories(log_dir)

    # Filter logs
    filtered_logs = []
    
    # Normalize inputs for comparison
    algo_set = set(a.lower() for a in algorithms) if algorithms else None
    target_env = environment.lower() if environment else None

    for log_path in all_logs:
        info = parse_log_name(log_path)
        
        log_algo = info["algorithm"].lower()
        log_env = info["environment"].lower()

        if algo_set and log_algo not in algo_set:
            continue
        if target_env and target_env not in log_env: # loose match for env name
             # Try exact match if loose match fails?
             # Usually env name in log is "Walker2d-v4", target might be "Walker2d"
             pass
        elif target_env and log_env != target_env:
            # If target_env is provided, check strict match or substring
            if target_env not in log_env:
                continue

        filtered_logs.append(log_path)

    if not filtered_logs:
        print(f"No matching logs found for algos={algorithms}, env={environment}")
        return None
        
    print(f"Found {len(filtered_logs)} matching logs.")

    return plot_rewards(
        log_paths=filtered_logs,
        save_path=save_path,
        show=show,
    )


def main():
    parser = argparse.ArgumentParser(description="Plot RL training reward curves")
    parser.add_argument("--log-dir", type=str, default="./log", help="Log directory")
    parser.add_argument("--save", type=str, default="./plots/rewards.png", help="Save path for the figure (default: ./plots/rewards.png)")
    parser.add_argument("--smooth", type=int, default=5, help="Smoothing window size")
    parser.add_argument("--algo", type=str, nargs="+", help="Algorithms to compare")
    parser.add_argument("--env", type=str, help="Environment to filter")
    parser.add_argument("--list", action="store_true", help="List available logs")
    parser.add_argument("--no-show", action="store_true", help="Don't show plot window")

    args = parser.parse_args()
    
    # Set backend to Agg if no-show is requested to avoid GUI issues
    if args.no_show:
        matplotlib.use('Agg')

    if args.list:
        logs = find_log_directories(args.log_dir)
        print(f"\nFound {len(logs)} log directories:")
        for log in logs:
            info = parse_log_name(log)
            print(f"  - {info['algorithm']} / {info['environment']} ({log})")
        return

    if args.algo or args.env:
        compare_runs(
            log_dir=args.log_dir,
            algorithms=args.algo,
            environment=args.env,
            save_path=args.save,
            show=not args.no_show,
        )
    else:
        plot_rewards(
            log_dir=args.log_dir,
            save_path=args.save,
            smooth_window=args.smooth,
            show=not args.no_show,
        )


if __name__ == "__main__":
    main()
