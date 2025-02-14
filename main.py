import numpy as np
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

SEED = 42

# Register the custom environment
register(
    id='dam-v0',
    entry_point='env.dam:Dam',  
)

# Create the water reservoir environment

if __name__ == "__main__":
    config_name = "config_test"  # Change to "config_2" for the other config
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
        seed=config['seed']
    )

    print(len(algo.archive.individuals))

    # Train the algorithm using parameters from the selected configuration
    algo.train(
        total_timesteps=config['total_timesteps'],
        eval_env=make_env(config['env_id'], config['seed'], 1, "PGMORL_eval_env", config['gamma'])(),
        ref_point=config['ref_point'],
        known_pareto_front=None,
    )
    
    env = make_env(config['env_id'], (SEED), 1, "PGMORL_test", gamma=0.995)()  # idx != -1 to avoid taking videos

    print(len(algo.archive.individuals))
    all_rewards = []
    # Execution of trained policies
    for a in algo.archive.individuals:
        #scalarized, discounted_scalarized, reward, discounted_reward = eval_mo(
        #    agent=a, env=env, w=np.array([1.0, 1.0]), render=False
        #)
        scalarized, discounted_scalarized, reward, discounted_reward = a.policy_eval(env, num_episodes=5, scalarization=np.dot, weights=np.array([1.0, 1.0]))
        print(f"Agent #{a.id}")
        print(f"Scalarized: {scalarized}")
        print(f"Discounted scalarized: {discounted_scalarized}")
        print(f"Vectorial: {reward}")
        print(f"Discounted vectorial: {discounted_reward}")
        print(f"Weights: {a.np_weights}")
        all_rewards.append(reward)  # Store reward vector


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
    