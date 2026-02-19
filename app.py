"""
RL Training Framework - Web UI
A clean, modern interface for training and evaluating RL agents.
"""

import streamlit as st
import os
import time
import subprocess
import glob
from pathlib import Path
from typing import Optional, Dict, Any
import json
import textwrap
import torch
import sys
import re
import pandas as pd

from src.ui_components import card, metric_card, section_header, feature_list

# Page configuration
st.set_page_config(
    page_title="RL Training Framework",
    page_icon="assets/icon_home.svg", # Use SVG if supported or emoji
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
def load_css():
    css_path = Path("assets/style.css")
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "training_started" not in st.session_state:
    st.session_state.training_started = False
if "training_process" not in st.session_state:
    st.session_state.training_process = None
if "training_complete" not in st.session_state:
    st.session_state.training_complete = False

# Helper function for navigation
def navigate_to(page_name):
    st.session_state.page = page_name

# ============== Sidebar ==============
with st.sidebar:
    st.markdown("###  RL Training")
    st.markdown("---")

    # Define pages with clean names
    pages = {
        "Home": "Home", 
        "Train": "Train",
        "Test": "Test", 
        "Visualize": "Visualize", 
        "Settings": "Settings"
    }

    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # Sync selection with session state
    if st.session_state.page in pages:
        index = list(pages.keys()).index(st.session_state.page)
    else:
        index = 0

    selection = st.radio(
        "Navigation",
        list(pages.keys()),
        label_visibility="collapsed",
        index=index,
    )
    
    # Update page if radio button changed
    if selection != st.session_state.page:
        st.session_state.page = selection
        st.rerun()

    page = st.session_state.page

    st.markdown("---")

    # Quick stats
    st.markdown("##### Quick Stats")
    log_dirs = glob.glob("log/*")
    checkpoints = glob.glob("checkpoints/*.pth")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Runs", len(log_dirs))
    with col2:
        st.metric("Models", len(checkpoints))


# ============== Home Page ==============
if page == "Home":
    st.markdown('<div style="text-align: center; margin-bottom: 2rem;">', unsafe_allow_html=True)
    st.markdown('# RL Training Framework')
    st.markdown('<p style="font-size: 1.2rem; color: var(--text-sub);">Train, evaluate, and visualize reinforcement learning agents with ease</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Feature cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        card("Train", "Train agents with DQN, PPO, SAC, or TD3")
        if st.button("Go to Train", use_container_width=True, key="btn_train"):
            st.session_state.page = "Train"
            st.rerun()
            
    with col2:
        card("Test", "Evaluate trained models")
        if st.button("Go to Test", use_container_width=True, key="btn_test"):
            st.session_state.page = "Test"
            st.rerun()
            
    with col3:
        card("Visualize", "Plot reward curves")
        if st.button("Go to Visualize", use_container_width=True, key="btn_viz"):
            st.session_state.page = "Visualize"
            st.rerun()
            
    with col4:
        card("Configure", "Custom settings")
        if st.button("Go to Settings", use_container_width=True, key="btn_settings"):
            st.session_state.page = "Settings"
            st.rerun()

    st.markdown("---")

    # Available environments
    section_header(" Available Environments")

    env_cols = st.columns(2)

    with env_cols[0]:
        feature_list("Discrete Action Space", [
            {"name": "CartPole-v1", "desc": "Balance a pole on a cart"},
            {"name": "MountainCar-v0", "desc": "Drive a car up a hill"},
            {"name": "Acrobot-v1", "desc": "Swing a robot arm"}
        ])

    with env_cols[1]:
        feature_list("Continuous Action Space", [
            {"name": "HalfCheetah-v4", "desc": "Run with a cheetah robot"},
            {"name": "Walker2d-v4", "desc": "Walk with a bipedal robot"},
            {"name": "Hopper-v4", "desc": "Hop with a single-legged robot"},
            {"name": "Ant-v4", "desc": "Walk with a quadrupedal robot"}
        ])

    # Quick start
    st.markdown("---")
    section_header(" Quick Start")
    st.code("""
# Start training with just a few clicks!
1. Go to the Train tab
2. Select environment and algorithm
3. Click 'Start Training'
4. Watch the results in real-time
    """, language="bash")


# ============== Train Page ==============
elif page == "Train":
    section_header("Train Agent", "Configure and start training your RL agent")

    train_col1, train_col2 = st.columns([2, 1])

    with train_col1:
        st.markdown("#### Configuration")
        
        # Algorithm selection
        algo = st.selectbox(
            "Algorithm",
            ["ppo", "dqn", "sac", "td3"],
            format_func=lambda x: x.upper(),
            help="Select the RL algorithm to use"
        )

        # Environment selection based on algorithm
        if algo in ["dqn", "ppo"]:
            env_options = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"]
            st.info("💡 Discrete algorithms work with discrete action spaces")
        else:
            env_options = ["HalfCheetah-v4", "Walker2d-v4", "Hopper-v4", "Ant-v4"]
            st.info("💡 Continuous algorithms work with continuous action spaces")

        env = st.selectbox("Environment", env_options)

        # Training parameters
        st.markdown("#### Training Parameters")

        col1, col2, col3 = st.columns(3)

        with col1:
            epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=100, step=10)
        with col2:
            lr = st.select_slider("Learning Rate", options=[0.0001, 0.0003, 0.001, 0.003], value=0.0003)
        with col3:
            seed = st.number_input("Seed", min_value=0, max_value=9999, value=42)

        # Device selection
        cuda_available = torch.cuda.is_available()
        device_options = ["auto"]
        if cuda_available:
            device_options.append("cuda")
            st.success(f"GPU Available: {torch.cuda.get_device_name(0)}")
        device_options.append("cpu")

        device = st.selectbox("Device", device_options)

    with train_col2:
        # Calculate estimated time
        if "CartPole" in env:
            est_time = epochs * 3
        elif "Cheetah" in env or "Walker" in env:
            est_time = epochs * 15
        else:
            est_time = epochs * 10
            
        time_str = f"~{est_time:.0f} seconds" if est_time < 60 else f"~{est_time/60:.1f} minutes"

        summary_content = f"""<div style="margin-top: 1rem;"><p><strong>Algorithm</strong>: {algo.upper()}</p><p><strong>Environment</strong>: {env}</p><p><strong>Epochs</strong>: {epochs}</p><p><strong>Learning Rate</strong>: {lr}</p><p><strong>Seed</strong>: {seed}</p><p><strong>Device</strong>: {device}</p><hr style="margin: 12px 0; border-color: var(--border-color);"><p><strong>Est. Time</strong>: {time_str}</p></div>"""
        card("Summary", summary_content)

    # Start button
    st.markdown("---")

    if st.button(" Start Training", use_container_width=True, type="primary"):
        # Generate config file with algorithm-specific parameters
        sac_params = ""
        if algo == "sac":
            sac_params = f"""
  sac_alpha: 0.2
  sac_alpha_lr: {lr}
  sac_policy_lr: {lr}
  sac_qf_lr: {lr}
  sac_tau: 0.005
  sac_automatic_alpha_tuning: true
  sac_target_entropy: null
  sac_target_update_mode: "soft"
"""
        elif algo == "td3":
            sac_params = f"""
  td3_actor_lr: {lr}
  td3_critic_lr: {lr}
  td3_tau: 0.005
  td3_policy_noise: 0.2
  td3_noise_clip: 0.5
  td3_policy_freq: 2
"""
        elif algo == "ppo":
            sac_params = f"""
  ppo_clip_ratio: 0.2
  ppo_value_coef: 0.5
  ppo_ent_coef: 0.01
  ppo_max_grad_norm: 0.5
  ppo_gae_lambda: 0.95
  ppo_repeat_per_collect: 2
"""
        elif algo == "dqn":
            sac_params = f"""
  n_step: 3
  target_update_freq: 320
  eps_train: 0.1
  eps_test: 0.05
  is_double: true
"""
        else:
            sac_params = ""

        config_content = f"""# Auto-generated configuration
env:
  name: "{env}"
  train_num: 10
  test_num: 100

network:
  hidden_sizes: [256, 256]
  activation: "relu"

algorithm:
  name: "{algo}"
  gamma: 0.99
  lr: {lr}{sac_params}

training:
  epoch: {epochs}
  step_per_epoch: 10000
  step_per_collect: 2000
  episode_per_test: 100
  batch_size: 64

device:
  device: "{device}"

seed: {seed}
log_dir: "./log"
checkpoint_dir: "./checkpoints"
"""

        config_path = f"configs/auto_{algo}_{env.replace('-v', '')}.yaml"
        os.makedirs("configs", exist_ok=True)

        with open(config_path, "w") as f:
            f.write(config_content)

        # Store training info
        st.session_state.config_path = config_path
        st.session_state.algo = algo
        st.session_state.env = env

        # Start training
        st.session_state.training_started = True
        st.session_state.training_complete = False

        # Create output area
        output_area = st.empty()
        
        # Metrics area
        metrics_placeholder = st.empty()
        
        # Logs area
        log_expander = st.expander("Terminal Output", expanded=True)
        with log_expander:
            log_text = st.empty()

        progress_bar = st.progress(0)

        with output_area:
            st.info("🔄 Training started... This may take a while.")
            st.code(f"Command: python train.py train --config {config_path}")

        # Run training in background
        try:
            cmd = ["python", "-m", "src.main", "train", "--config", config_path]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            log_lines = []
            epoch_metrics = []
            
            # Regex patterns
            # Pattern for progress bar: "Epoch #1: 100%|...| ... rew=10.00, len=200.00, loss=0.123 ..."
            # Note: Tianshou format might vary slightly (e.g. PPO might not show loss in progress bar)
            pat_progress = re.compile(r"Epoch #(\d+): .*?rew=([-\d\.]+)(?:.*?len=([-\d\.]+))?(?:.*?loss=([-\d\.]+))?")
            
            # Pattern for test results: "Epoch #1: test_reward: 123.45 ± 10.00, best_reward: ..."
            pat_test = re.compile(r"Epoch #(\d+): test_reward: ([-\d\.]+).*?best_reward: ([-\d\.]+)")

            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                    
                log_lines.append(line)
                
                # Keep last 1000 lines to avoid memory issues but show enough context
                if len(log_lines) > 1000:
                    log_lines = log_lines[-1000:]
                
                # Update log text (show last 30 lines for scrolling effect)
                log_text.code("\n".join(log_lines[-30:]), language="bash")

                # Parse metrics
                # Check for test results first (usually at end of epoch)
                test_match = pat_test.search(line)
                if test_match:
                    epoch = int(test_match.group(1))
                    test_reward = float(test_match.group(2))
                    best_reward = float(test_match.group(3))
                    
                    # Update or append metric
                    # Check if we already have this epoch (from progress bar)
                    existing = next((item for item in epoch_metrics if item["Epoch"] == epoch), None)
                    if existing:
                        existing["Test Reward"] = test_reward
                        existing["Best Reward"] = best_reward
                    else:
                        epoch_metrics.append({
                            "Epoch": epoch,
                            "Test Reward": test_reward,
                            "Best Reward": best_reward,
                            "Train Reward": None, # Placeholder
                            "Loss": None
                        })
                    
                    # Update dataframe
                    df = pd.DataFrame(epoch_metrics).set_index("Epoch")
                    metrics_placeholder.dataframe(df.sort_index(ascending=False).head(5), use_container_width=True)

                # Check for progress bar updates
                prog_match = pat_progress.search(line)
                if prog_match:
                    epoch = int(prog_match.group(1))
                    rew = float(prog_match.group(2))
                    length = float(prog_match.group(3)) if prog_match.group(3) else 0.0
                    loss = float(prog_match.group(4)) if prog_match.group(4) else 0.0
                    
                    # Update progress bar
                    progress = min(epoch / epochs, 1.0)
                    progress_bar.progress(progress)
                    
                    # Update metrics
                    existing = next((item for item in epoch_metrics if item["Epoch"] == epoch), None)
                    if existing:
                        existing["Train Reward"] = rew
                        existing["Loss"] = loss
                    else:
                        epoch_metrics.append({
                            "Epoch": epoch,
                            "Test Reward": None,
                            "Best Reward": None,
                            "Train Reward": rew,
                            "Loss": loss
                        })
                    
                    # Update dataframe
                    df = pd.DataFrame(epoch_metrics).set_index("Epoch")
                    metrics_placeholder.dataframe(df.sort_index(ascending=False).head(5), use_container_width=True)

            process.wait()

            if process.returncode == 0:
                st.session_state.training_complete = True
                st.success("✅ Training completed successfully!")
                progress_bar.progress(1.0)
            else:
                st.error(f"❌ Training failed with exit code {process.returncode}")

        except Exception as e:
            st.error(f"Error during training: {e}")

    # Show training history
    st.markdown("---")
    st.markdown("#### Recent Training Runs")

    log_dirs = sorted(glob.glob("log/*"), key=os.path.getmtime, reverse=True)[:5]

    if log_dirs:
        for log_dir in log_dirs:
            dir_name = os.path.basename(log_dir)
            parts = dir_name.split("_")
            if len(parts) >= 2:
                algo_name = parts[0]
                env_name = "_".join(parts[1:-1]) if len(parts) > 2 else parts[1]

                col1, col2, col3 = st.columns([2, 2, 1])
                col1.write(f"**{algo_name.upper()}** / {env_name}")
                col2.write(f"`{log_dir}`")
                col3.write(time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(log_dir))))
    else:
        st.info("No training runs yet. Start your first training above!")


# ============== Test Page ==============
elif page == "Test":
    section_header("Test Model", "Evaluate a trained RL agent")

    # Find available models
    checkpoint_files = glob.glob("checkpoints/*.pth")

    if not checkpoint_files:
        st.warning("⚠️ No trained models found. Please train a model first.")
        st.markdown("""
        Models are saved in the `checkpoints/` directory during training.
        The best model is typically saved as `<algo>_<env>_best.pth`.
        """)
    else:
        test_col1, test_col2 = st.columns([2, 1])

        with test_col1:
            st.markdown("#### Configuration")
            
            # Model selection
            model_options = {}
            for f in checkpoint_files:
                basename = os.path.basename(f)
                model_options[basename] = f

            selected_model = st.selectbox(
                "Trained Model",
                list(model_options.keys()),
                format_func=lambda x: f" {x}"
            )

            # Extract env from model name
            model_name = selected_model.replace(".pth", "")
            parts = model_name.split("_")

            # Auto-detect environment
            if len(parts) >= 2:
                detected_env = parts[1] if len(parts) == 2 else "_".join(parts[1:-1])
                # Reconstruct full env name
                if detected_env not in ["CartPole", "MountainCar", "Acrobot", "HalfCheetah", "Walker2d", "Hopper", "Ant"]:
                    env_name = detected_env
                else:
                    env_name = detected_env + "-v1" if detected_env in ["CartPole", "MountainCar", "Acrobot"] else detected_env + "-v4"
            else:
                env_name = "CartPole-v1"

            env = st.text_input("Environment", value=env_name)

            # Test parameters
            st.markdown("#### Test Parameters")

            col1, col2 = st.columns(2)

            with col1:
                episodes = st.number_input("Episodes", min_value=1, max_value=1000, value=10)
            with col2:
                render = st.checkbox("Render", value=False, help="Show environment visualization")

            algo = st.selectbox("Algorithm", ["ppo", "dqn", "sac", "td3"], format_func=lambda x: x.upper())

        with test_col2:
            summary_content = f"""<div style="margin-top: 1rem;"><p><strong>Model</strong>: `{selected_model}`</p><p><strong>Environment</strong>: {env}</p><p><strong>Episodes</strong>: {episodes}</p><p><strong>Render</strong>: {'Yes' if render else 'No'}</p></div>"""
            card("Summary", summary_content)

        # Test button
        st.markdown("---")

        if st.button(" Run Test", use_container_width=True, type="primary"):
            model_path = model_options[selected_model]

            output_area = st.empty()

            with output_area:
                st.info("🔄 Running evaluation...")

            import sys
            script_dir = os.path.dirname(os.path.abspath(__file__))

            cmd = [
                sys.executable, "-m", "src.main", "test",
                "--env", env,
                "--algo", algo,
                "--model", model_path,
                "--episodes", str(episodes),
            ]

            if render:
                cmd.append("--render")

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)

            # Parse output
            output = result.stdout + result.stderr

            # Extract results
            if "Average reward:" in output:
                for line in output.split("\n"):
                    if "Average reward:" in line:
                        st.success(line.strip())
                    elif "Min reward:" in line:
                        st.info(line.strip())
                    elif "Max reward:" in line:
                        st.info(line.strip())
            else:
                st.code(output[-1000:])


# ============== Visualize Page ==============
elif page == "Visualize":
    section_header("Visualize Results", "Plot and analyze training reward curves")

    # Find available logs
    log_dirs = glob.glob("log/*")

    if not log_dirs:
        st.warning("⚠️ No training logs found. Please train a model first.")
    else:
        # Parse log info
        log_info = []
        for log_dir in log_dirs:
            dir_name = os.path.basename(log_dir)
            parts = dir_name.split("_")

            if len(parts) >= 2:
                algo = parts[0]
                env = "_".join(parts[1:-1]) if len(parts) > 2 else parts[1]
            else:
                algo = "unknown"
                env = "unknown"

            log_info.append({
                "path": log_dir,
                "name": dir_name,
                "algo": algo,
                "env": env,
                "time": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(log_dir))),
            })

        # Filter options
        st.markdown("#### Filters")
        
        # Use card for filters to group them nicely
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                algos = list(set(info["algo"] for info in log_info))
                selected_algos = st.multiselect("Algorithms", algos, default=algos)

            with col2:
                envs = list(set(info["env"] for info in log_info))
                selected_envs = st.multiselect("Environments", envs, default=envs)

        # Filter logs
        filtered_logs = [
            info for info in log_info
            if info["algo"] in selected_algos and info["env"] in selected_envs
        ]

        st.markdown(f"Found **{len(filtered_logs)}** matching runs")

        # Display log table
        if filtered_logs:
            st.markdown("#### Training Runs")

            # Create a clean layout for log entries
            for info in filtered_logs:
                # Use columns to align the expander and icon better if needed,
                # but standard st.expander usually handles this well.
                # If icons are overlapping text, it might be a Streamlit rendering issue with emojis.
                # We'll simplify the label to be safe.
                label = f"{info['algo'].upper()} | {info['env']} | {info['time']}"
                with st.expander(label):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Directory**: `{info['path']}`")
                        st.write(f"**Algorithm**: {info['algo'].upper()}")
                    with col2:
                        st.write(f"**Environment**: {info['env']}")
                        st.write(f"**Date**: {info['time']}")

                    # Show if there's a plot for this run
                    plot_files = glob.glob(f"plots/*{info['name']}*.png")
                    if plot_files:
                        st.success(f"📈 Plot exists: `{plot_files[0]}`")
                        try:
                            st.image(plot_files[0], width=400)
                        except:
                            pass
                    else:
                        st.info("No plot generated yet")

        # Generate comparison plot
        st.markdown("---")
        section_header("Generate Comparison Plot")
        
        # File Browser for specific file selection
        with st.expander("📂 Advanced File Selection"):
            st.markdown("Browse and select specific log files to plot.")
            
            # Base directory selector
            base_dir = st.text_input("Root Directory", value="./log")
            
            selected_file = None
            if os.path.exists(base_dir):
                # Walk through directory to find event files
                event_files = []
                for root, dirs, files in os.walk(base_dir):
                    for file in files:
                        if "events.out.tfevents" in file:
                            full_path = os.path.join(root, file)
                            # Get relative path for display
                            rel_path = os.path.relpath(full_path, base_dir)
                            event_files.append((rel_path, full_path))
                
                if event_files:
                    file_options = {p[0]: p[1] for p in event_files}
                    selected_rel_path = st.selectbox(
                        "Select Log File", 
                        list(file_options.keys()),
                        format_func=lambda x: f"📄 {x}"
                    )
                    
                    if selected_rel_path:
                        selected_file = file_options[selected_rel_path]
                        
                        # Show file info
                        file_stat = os.stat(selected_file)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption(f"Size: {file_stat.st_size / 1024:.2f} KB")
                        with col2:
                            st.caption(f"Modified: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_stat.st_mtime))}")
                else:
                    st.info("No event files found in this directory.")
            else:
                st.error("Directory not found.")

        if len(filtered_logs) >= 1 or selected_file:
            col1, col2, col3 = st.columns(3)

            with col1:
                smooth = st.slider("Smoothing", 5, 50, 10)
            with col2:
                st.write(" ")  # Spacer
            with col3:
                generate = st.button("📈 Generate Plot", type="primary", use_container_width=True)

            if generate:
                import sys
                # Auto-generate unique filename with timestamp
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_name = f"plots/comparison_{timestamp}.png"

                abs_save_path = os.path.abspath(save_name)
                abs_plots_dir = os.path.dirname(abs_save_path)
                os.makedirs(abs_plots_dir, exist_ok=True)

                cmd = [
                    sys.executable, "-m", "src.plot_rewards" if os.path.exists("src/plot_rewards.py") else "plot_rewards",
                    "--smooth", str(smooth),
                    "--save", abs_save_path,
                    "--no-show",
                ]
                
                # If a specific file is selected in advanced mode, use it
                if selected_file:
                    cmd.extend(["--log-dir", selected_file])
                else:
                    # Otherwise use the filtered logs from the standard view
                    # Add algo filter if specific algos selected
                    if selected_algos:
                        for algo in selected_algos:
                            cmd.extend(["--algo", algo])

                    # Add env filter if single env selected
                    if len(selected_envs) == 1:
                        cmd.extend(["--env", selected_envs[0]])

                script_dir = os.path.dirname(os.path.abspath(__file__))
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)

                if result.returncode == 0:
                    import time as time_module
                    time_module.sleep(0.3)
                    if os.path.exists(abs_save_path):
                        st.success(f"Plot generated: `{save_name}`")
                        st.image(abs_save_path, width=800)
                    else:
                        st.error(f"File not found: {abs_save_path}")
                else:
                    st.error(f"Command failed: {result.stderr}")
        else:
            st.info("Select at least one log or use the file browser to generate a plot")


# ============== Settings Page ==============
elif page == "Settings":
    section_header("Settings", "Configure the RL Framework")

    # System info
    st.markdown("#### System Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card("Python", "3.x")
    with col2:
        metric_card("PyTorch", torch.__version__)
    with col3:
        metric_card("CUDA", "Available" if torch.cuda.is_available() else "N/A")

    if torch.cuda.is_available():
        st.success(f"GPU: {torch.cuda.get_device_name(0)}")

    st.markdown("---")

    # Quick config generation
    section_header("Generate Configuration File")

    col1, col2 = st.columns(2)

    with col1:
        config_env = st.selectbox(
            "Environment",
            ["CartPole-v1", "MountainCar-v0", "HalfCheetah-v4", "Walker2d-v4"]
        )
    with col2:
        config_algo = st.selectbox("Algorithm", ["ppo", "dqn", "sac", "td3"])

    if st.button("📄 Generate Config", use_container_width=True):
        config_content = f"""# Configuration for {config_algo.upper()} on {config_env}
env:
  name: "{config_env}"
  train_num: 10
  test_num: 100

network:
  hidden_sizes: [256, 256]
  activation: "relu"

algorithm:
  name: "{config_algo}"
  gamma: 0.99
  lr: 0.0003

training:
  epoch: 100
  step_per_epoch: 10000
  step_per_collect: 2000
  episode_per_test: 100
  batch_size: 64

device:
  device: "auto"

seed: 42
log_dir: "./log"
checkpoint_dir: "./checkpoints"
"""
        output_file = f"configs/{config_algo}_{config_env.replace('-v', '')}.yaml"

        st.download_button(
            label="📥 Download Config",
            data=config_content,
            file_name=output_file,
            mime="text/yaml",
            use_container_width=True,
        )

    st.markdown("---")

    # Data management
    st.markdown("#### Data Management")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Logs")
        log_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk("log")
            for filename in filenames
        )
        st.write(f"Total size: {log_size / 1024 / 1024:.2f} MB")

        if st.button("🗑️ Clear Logs", key="clear_logs", use_container_width=True):
            import shutil
            if os.path.exists("log"):
                shutil.rmtree("log")
                os.makedirs("log")
                st.success("Logs cleared!")
                st.rerun()

    with col2:
        st.markdown("##### Checkpoints")
        checkpoint_size = sum(
            os.path.getsize(f) for f in glob.glob("checkpoints/*.pth")
        )
        st.write(f"Total size: {checkpoint_size / 1024 / 1024:.2f} MB")

        if st.button("🗑️ Clear Checkpoints", key="clear_checkpoints", use_container_width=True):
            import shutil
            if os.path.exists("checkpoints"):
                for f in glob.glob("checkpoints/*.pth"):
                    os.remove(f)
                st.success("Checkpoints cleared!")
                st.rerun()


# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: var(--text-sub); padding: 1rem;">
        <p>RL Training Framework | Created by <strong>moying</strong></p>
        <p style="font-size: 0.8rem;">Built with Streamlit • Supports DQN, PPO, SAC, and TD3</p>
    </div>
    """,
    unsafe_allow_html=True
)
