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
from igmorl import IGMORL, make_env
from config import *
from e_nautilus import E_NAUTILUS

SEED = 42
SAVE = False
E_NAUT = True

# Register the custom environment
register(
    id='dam-v0',
    entry_point='env.dam:Dam',  
)

# Create the water reservoir environment

if __name__ == "__main__":
    config_name = "config_1"  # Change to "config_2" for the other config
    config = load_config(config_name)

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
        log=False,
        seed=config['seed'],
        interactive=False,
    )

    # Train the algorithm using parameters from the selected configuration
    algo.train(
        total_timesteps=config['total_timesteps'],
        eval_env=make_env(config['env_id'], config['seed'], 1, "PGMORL_eval_env", config['gamma'])(),
        ref_point=config['ref_point'],
        known_pareto_front=None,
    )
    
    env = make_env(config['env_id'], (config['seed']-1), 1, "PGMORL_test", gamma=0.995)()  # idx != -1 to avoid taking videos

    print(len(algo.archive.individuals))
    

    all_rewards = []
    # Initialize a list to store the data for each agent
    agent_data = []

    # Visualization of trained policies
    for (a, e) in zip(algo.archive.individuals, algo.archive.evaluations):
        # Evaluate policy
        #scalarized, discounted_scalarized, reward, discounted_reward = a.policy_eval(env, num_episodes=5, scalarization=np.dot, weights=np.array([1.0, 1.0]))
        
        # Prepare agent data
        agent_info = {
            'Experiment': "interactive-dim_2-config_1",
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
        e_nautilus = E_NAUTILUS(all_rewards)

        # Run the interactive method
        selected_solution = e_nautilus.run()

        print("Final Selected Solution:", selected_solution)

    