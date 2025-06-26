import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.envs.registration import register
from igmorl.igmorl import IGMORL, make_env
from igmorl.e_nautilus import E_NAUTILUS
from matplotlib.animation import FuncAnimation
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch as th
import random
import time
import datetime

# Add the parent directory of 'examples' to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1] / "env" / "wms_morl-main"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "examples"))

def user_utility_focus_a(a, b):
    return a * 0.7 + b * 0.3

def user_utility_focus_b(a, b):
    return a * 0.3 + b * 0.7

def user_utility_even(a, b):
    return a * 0.5 + b * 0.5

def make_normalized_utility(utility_func, min_a, max_a, min_b, max_b):
    def normalized_utility(a, b):
        norm_a = (a - min_a) / (max_a - min_a) if max_a > min_a else 0.0
        norm_b = (b - min_b) / (max_b - min_b) if max_b > min_b else 0.0
        return utility_func(norm_a, norm_b)
    return normalized_utility

# Map utility function names to their implementations
UTILITY_FUNCTIONS = {
    "focus_a": user_utility_focus_a,
    "focus_b": user_utility_focus_b,
    "even": user_utility_even,
}

# Define command-line arguments
parser = argparse.ArgumentParser(description="Run the MORL algorithm with configurable parameters.")
parser.add_argument("--cfg", type=str, default="car", help="Configuration to use (e.g., nile, cheetah).")
parser.add_argument("--i", action="store_true", help="Enable interactive mode.")
parser.add_argument("--save", action="store_true", help="Enable saving results.")
parser.add_argument("--log", type=bool, default=False, help="Enable or disable logging (True/False).")
parser.add_argument("--has_t", action="store_true", help="Enable target checking.")
parser.add_argument("--t", type=str, default="[100, 100]", help="Target for the Nile River simulation (e.g., [-1, -1.5]).")
parser.add_argument("--a", action="store_true", help="Enable artificial user selection.")
parser.add_argument("--out", type=str, default="agent_performance.csv", help="Output file name for agent performance data.")
parser.add_argument("--exp", type=str, default="no_interactive", help="Experiment type (e.g., no_interactive, interactive).")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
parser.add_argument("--norm", action="store_true", help="Enable normalization of utility function.")
parser.add_argument("--uu", type=str, default="even", choices=UTILITY_FUNCTIONS.keys(), help="User utility function to use (e.g., focus_a, focus_b, even).")

# Parse the arguments
args = parser.parse_args()

# Convert the target argument (string) to a NumPy array
if isinstance(args.t, str):
    args.t = np.array(eval(args.t))  # Safely evaluate the string to a list and convert to np.ndarray

# Assign the parsed arguments to variables
CONFIG = args.cfg
INTERACTIVE = args.i
SAVE = args.save
LOG = args.log
HAS_TARGET = args.has_t
ARTIFICIAL = args.a
TARGET = args.t  # Now correctly parsed as a NumPy array
EXPERIMENT = args.exp
OUTPUT = args.out
SEED = args.seed
NORM = args.norm

E_NAUT = False

# Retrieve the selected utility function
user_utility = UTILITY_FUNCTIONS[args.uu]
if NORM:
    # Define the normalization parameters based on the configuration
    if CONFIG == "cheetah":
        min_a, max_a = -100, 10  # Example values for Cheetah River simulation
        min_b, max_b = -100, 0  # Example values for Cheetah River simulation
    elif CONFIG == "car":
        min_a, max_a = -110, 0  # Example values for Car environment
        min_b, max_b = -110, 0  # Example values for Car environment
    elif CONFIG == "swimmer":
        min_a, max_a = -100, 10
        min_b, max_b = -100, 0
    else:
        raise ValueError(f"Unknown configuration: {CONFIG}")

    # Wrap the user utility function with normalization
    user_utility = make_normalized_utility(user_utility, min_a, max_a, min_b, max_b)

# Register the custom environment
register(
    id='dam-v0',
    entry_point='env.water_dam.dam:Dam',  
)

register(
    id='nile2-v1',
    entry_point='env.wms_morl-main.examples.nile_river_simulation:create_nile_river_env',
    kwargs={'custom_obj': ['ethiopia_power', 'egypt_deficit_minimised']}
)



def plot_pareto_progression(history, show=True, save_path="pareto_front_evolution.mp4"):
    """Visualizes how the Pareto front evolved during training and saves it as a video.
    
    Args:
        history: The history of Pareto fronts during training.
        show: Whether to display the plot immediately.
        save_path: The path to save the video file (default is 'pareto_front_evolution.mp4').
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get all unique Pareto points from history
    all_points = []
    for entry in history:
        all_points.extend(entry['pareto_front'])
    all_points = np.array(all_points)
    
    # Set plot limits
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1
    
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlabel('Objective 1')
    ax.set_ylabel('Objective 2')
    ax.set_title('Pareto Front Progression')
    
    # Initialize plot elements
    front_line, = ax.plot([], [], 'ro-', label='Current Pareto Front')
    ax.legend()
    
    # Define a colormap for different iterations
    cmap = plt.cm.get_cmap('viridis', len(history))
    iteration_colors = [cmap(i) for i in range(len(history))]
    
    def init():
        front_line.set_data([], [])
        return (front_line,)
    
    def update(frame):
        if frame < len(history):  # Show individual iterations
            data = np.array(history[frame]['pareto_front'])
            if len(data) > 1:
                # Sort points for better visualization
                sorted_idx = np.argsort(data[:, 0])
                data = data[sorted_idx]
            
            front_line.set_data(data[:, 0], data[:, 1])
            ax.set_title(f'Pareto Front (Iteration {history[frame]["iteration"]}, Step {history[frame]["global_step"]})')
            return (front_line,)
        else:  # Final frame: Show all iterations
            for i, entry in enumerate(history):
                data = np.array(entry['pareto_front'])
                if len(data) > 1:
                    # Sort points for better visualization
                    sorted_idx = np.argsort(data[:, 0])
                    data = data[sorted_idx]
                ax.plot(data[:, 0], data[:, 1], marker='o', linestyle='-', color=iteration_colors[i], label=f'Iteration {i+1}')
            ax.legend()
            ax.set_title('Final Pareto Front with All Iterations')
            return ()
    
    # Create the animation
    ani = FuncAnimation(
        fig,
        update,
        frames=len(history) + 1,  # Add one extra frame for the final frame
        init_func=init,
        blit=False,  # Set to False to allow multiple plots in the final frame
        interval=500,
        repeat_delay=2000
    )
    
    # Save the animation as a video
    if save_path:
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=2, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(save_path, writer=writer)
        print(f"Video saved to: {os.path.abspath(save_path)}")
    
    if show:
        plt.show()
    
    return ani

# Load the configuration from the JSON file
def load_config(config_name):
    config_path = os.path.join(os.getcwd(), "config.json")
    with open(config_path, "r") as file:
        configs = json.load(file)
    if config_name not in configs:
        raise ValueError(f"Unknown configuration: {config_name}")
    config = configs[config_name]

    # Convert lists to NumPy arrays where necessary
    config["ref_point"] = np.array(config["ref_point"])
    config["origin"] = np.array(config["origin"])
    return config

# Example usage
if __name__ == "__main__":
    # Set seeds for reproducibility
    th.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(SEED)

    config = load_config(CONFIG)

    # Initialize the algorithm using parameters from the selected configuration
    algo = IGMORL(
        env_id=config["env_id"],
        num_envs=config["num_envs"],
        pop_size=config["pop_size"],
        warmup_iterations=config["warmup_iterations"],
        evolutionary_iterations=config["evolutionary_iterations"],
        num_weight_candidates=config["num_weight_candidates"],
        origin=config["origin"],
        steps_per_iteration=config["steps_per_iteration"],
        delta_weight=config["delta_weight"],
        sparsity_coeff=config["sparsity_coeff"],
        log=LOG,
        experiment_name= EXPERIMENT,
        wandb_entity="henwei1998",
        seed=SEED,
        interactive=INTERACTIVE,
        artificial=ARTIFICIAL,
        user_utility=user_utility,
        has_target=HAS_TARGET,
        target=TARGET
    )

    # --- TIMER START ---
    start_time = time.time()

    # Train the algorithm using parameters from the selected configuration
    history, bounds = algo.train(
        total_timesteps=config["total_timesteps"],
        eval_env=make_env(
            config["env_id"], SEED, 1, "PGMORL_eval_env", config["gamma"]
        )(),
        ref_point=config["ref_point"],
        known_pareto_front=None,
    )

    # --- TIMER END ---
    elapsed_time = time.time() - start_time
    print(f"Training completed in {str(datetime.timedelta(seconds=int(elapsed_time)))} (hh:mm:ss)")

    env = make_env(config['env_id'], (SEED-1), 1, "PGMORL_test", gamma=0.995)()  # idx != -1 to avoid taking videos

    print(algo.archive.evaluations)
    print(f"Boundaries: {bounds}")
    
    # Save the video in the current directory
    save_path = os.path.join(os.getcwd(), "pareto_front_evolution.mp4")
    #plot_pareto_progression(history, show=True, save_path=save_path)
    

    all_rewards = []
    # Initialize a list to store the data for each agent
    agent_data = []

    # Visualization of trained policies
    for (a, e) in zip(algo.archive.individuals, algo.archive.evaluations):
        # Prepare agent data (no filtering based on bounds)
        agent_info = {
            'Experiment': EXPERIMENT,
            'Agent ID': a.id,
            'Vectorial Reward': ";".join(map(str, e)),  # Convert list to string for CSV
            'Weights': ";".join(map(str, a.np_weights.tolist())),  # Convert weights to string
            'Bounds': ";".join(map(str, bounds)),  # Add bounds as a string
        }

        # Append agent data to the list
        agent_data.append(agent_info)

        # Optionally print to the console (as in your original code)
        print(f"Agent #{a.id}")
        print(f"Vectorial: {e}")
        print(f"Weights: {a.np_weights}")
        print(f"Bounds: {bounds}")

    # Save results if there are any agents in the archive
    if SAVE and agent_data:
        # Convert the list of agent data into a DataFrame
        df = pd.DataFrame(agent_data)

        # Save the DataFrame to a CSV file
        file_exists = os.path.isfile(OUTPUT)

        # Append data to CSV (without writing headers if the file already exists)
        df.to_csv(OUTPUT, mode='a', index=False, header=not file_exists)

        print(f"Data has been written to {OUTPUT}")
    else:
        print("No agents met the criteria to save results.")

    if E_NAUT:
        df = pd.DataFrame(all_rewards, columns=["Reward 1", "Reward 2"])
        df.to_csv("all_rewards.csv", index=False)
        e_nautilus = E_NAUTILUS(all_rewards)

        # Run the interactive method
        selected_solution = e_nautilus.run()

        print("Final Selected Solution:", selected_solution)


