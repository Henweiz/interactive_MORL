import numpy as np

config_1 = {
    'env_id': "mo-mountaincarcontinuous-v0",
    'seed': 42,
    'ref_point': np.array([-110, -110]),
    'num_envs': 4,
    'pop_size': 7,
    'warmup_iterations': 10,
    'evolutionary_iterations': 5,
    'num_weight_candidates': 50,
    'origin': np.array([-110, -110]),
    'steps_per_iteration': 1000,
    'delta_weight': 0.2,
    'total_timesteps': 1500000,
    'gamma': 0.995
}

config_test = {
    'env_id': "mo-mountaincarcontinuous-v0",
    'seed': 42,
    'ref_point': np.array([-110, -110]),
    'num_envs': 4,
    'pop_size': 7,
    'warmup_iterations': 1,
    'evolutionary_iterations': 5,
    'num_weight_candidates': 50,
    'origin': np.array([-110, -110]),
    'steps_per_iteration': 100,
    'delta_weight': 0.2,
    'total_timesteps': 50000,
    'gamma': 0.995
}

# A function to load a configuration
def load_config(config_name):
    if config_name == "config_1":
        return config_1
    elif config_name == "config_test":
        return config_test
    else:
        raise ValueError(f"Unknown configuration: {config_name}")
