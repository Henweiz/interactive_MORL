import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
import mo_gymnasium as mo_gym
import morl_baselines as mb
#from morl_baselines.multi_policy.pgmorl.pgmorl import PGMORL
#from morl_baselines.single_policy.ser.mo_ppo import make_env
#from morl_baselines.common.evaluation import eval_mo
from gymnasium.envs.registration import register
from igmorl.igmorl import IGMORL, make_env
from config import *
from igmorl.e_nautilus import E_NAUTILUS
from matplotlib.animation import FuncAnimation

SEED = 42
SAVE = True
E_NAUT = False
LOG = False
INTERACTIVE = True
EXPERIMENT = "interactive_2_no_delta"  # Change to "interactive" for interactive agents
CONFIG = "config_2"  # Change to "config_test" for testing

# Register the custom environment
register(
    id='dam-v0',
    entry_point='env.dam:Dam',  
)

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

def run_e_naut():
    df = pd.read_csv('agent_performance.csv', header=None, names=['type', 'agent_id', 'vectorial', 'weights'])

    # Debug: Print the first few rows of the DataFrame to inspect the 'vectorial' column
    print("Original DataFrame:")
    print(df.head())

    # Convert the 'vectorial' column from strings to lists of floats
    df['vectorial'] = df['vectorial'].apply(lambda x: np.array(list(map(float, x.strip("[]").replace(" ", "").split(";")))))

    # Debug: Print the processed 'vectorial' column to ensure proper conversion
    print("Processed 'vectorial' column:")
    print(df['vectorial'].head())

    # Filter for non-interactive agents
    non_interactive = df[df['type'] == 'no_interactive']

    # Convert the 'vectorial' column to a 2D NumPy array
    vectorial_array = np.vstack(non_interactive['vectorial'].values)

    # Debug: Print the resulting 2D NumPy array
    print("2D NumPy array:")
    print(vectorial_array)

    # Pass the 2D array to E_NAUTILUS
    e_nautilus = E_NAUTILUS(vectorial_array)

    selected_solution = e_nautilus.run()
    print("Final Selected Solution:", selected_solution)
    return selected_solution

if __name__ == "__main__":
    #run_e_naut()

    config = load_config(CONFIG)

    # Initialize the algorithm using parameters from the selected configuration
    algo = IGMORL(
        env_id=config['env_id'],
        num_envs=config['num_envs'],
        pop_size=config['pop_size'],
        warmup_iterations=config['warmup_iterations'],
        evolutionary_iterations=config['evolutionary_iterations'],
        num_weight_candidates=config['num_weight_candidates'],
        origin=config['origin'],
        steps_per_iteration=config['steps_per_iteration'],
        delta_weight=config['delta_weight'],
        log=LOG,
        seed=config['seed'],
        interactive=INTERACTIVE,
        target=np.array([-0.8, -0.8])  # Target for the interactive agents
    )

    # Train the algorithm using parameters from the selected configuration
    history = algo.train(
        total_timesteps=config['total_timesteps'],
        eval_env=make_env(config['env_id'], config['seed'], 1, "PGMORL_eval_env", config['gamma'])(),
        ref_point=config['ref_point'],
        known_pareto_front=None,
    )
    
    env = make_env(config['env_id'], (config['seed']-1), 1, "PGMORL_test", gamma=0.995)()  # idx != -1 to avoid taking videos

    print(len(algo.archive.individuals))
    
    # Save the video in the current directory
    save_path = os.path.join(os.getcwd(), "pareto_front_evolution.mp4")
    plot_pareto_progression(history, show=True, save_path=save_path)
    

    all_rewards = []
    # Initialize a list to store the data for each agent
    agent_data = []

    # Visualization of trained policies
    for (a, e) in zip(algo.archive.individuals, algo.archive.evaluations):
        # Evaluate policy
        #scalarized, discounted_scalarized, reward, discounted_reward = a.policy_eval(env, num_episodes=5, scalarization=np.dot, weights=np.array([1.0, 1.0]))
        
        # Prepare agent data
        agent_info = {
            'Experiment': EXPERIMENT,
            'Agent ID': a.id,
            #'#Scalarized': scalarized,
            'Vectorial Reward': ";".join(map(str, e)),  # Convert list to string for CSV
            'Weights': ";".join(map(str, a.np_weights.tolist())),  # Convert weights to string
        }
        
        # Append agent data to the list
        agent_data.append(agent_info)

        # Optionally print to the console (as in your original code)
        print(f"Agent #{a.id}")
        #print(f"Scalarized: {scalarized}")
        print(f"Vectorial: {e}")
        print(f"Weights: {a.np_weights}")

        # Store the reward vector
        all_rewards.append(e)

    if SAVE:
        # Convert the list of agent data into a DataFrame
        df = pd.DataFrame(agent_data)

        # Save the DataFrame to an Excel file
        output_file = 'agent_performance.csv'
        # Check if file exists to avoid writing headers again
        file_exists = os.path.isfile(output_file)

        # Append data to CSV (without writing headers if the file already exists)
        df.to_csv(output_file, mode='a', index=False, header=not file_exists)

        print(f"Data has been written to {output_file}")

    # Convert to NumPy array for easy slicing
    all_rewards = np.array(all_rewards)

    # Scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(all_rewards[:, 0], all_rewards[:, 1], color='b', label="Agents")

    # Labels and title
    plt.xlabel("Reward Dimension 1")
    plt.ylabel("Reward Dimension 2")
    plt.title("Vectorial Rewards of Agents")
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()

    if E_NAUT:
        df = pd.DataFrame(all_rewards, columns=["Reward 1", "Reward 2"])
        df.to_csv("all_rewards.csv", index=False)
        e_nautilus = E_NAUTILUS(all_rewards)

        # Run the interactive method
        selected_solution = e_nautilus.run()

        print("Final Selected Solution:", selected_solution)


