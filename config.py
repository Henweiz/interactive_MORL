import numpy as np

config_1 = {
    'env_id': "mo-mountaincarcontinuous-v0",
    'seed': 42,
    'ref_point': np.array([-110, -110]),
    'num_envs': 4,
    'pop_size': 7,
    'warmup_iterations': 10,
    'evolutionary_iterations': 10,
    'num_weight_candidates': 50,
    'origin': np.array([-110, -110]),
    'steps_per_iteration': 1000,
    'delta_weight': 0.2,
    'total_timesteps': 2500000,
    'gamma': 0.995
}

config_2 = {
    'env_id': "mo-mountaincarcontinuous-v0",
    'seed': 42,
    'ref_point': np.array([-110, -110]),
    'num_envs': 4,
    'pop_size': 7,
    'warmup_iterations': 5,
    'evolutionary_iterations': 5,
    'num_weight_candidates': 50,
    'origin': np.array([-110, -110]),
    'steps_per_iteration': 1000,
    'delta_weight': 0.2,
    'total_timesteps': 1500000,
    'gamma': 0.995
}


config_cheetah = {
    'env_id': "mo-halfcheetah-v5",
    'seed': 42,
    'ref_point': np.array([-100, -100]),
    'num_envs': 4,
    'pop_size': 6,
    'warmup_iterations': 40,
    'evolutionary_iterations': 20,
    'num_weight_candidates': 50,
    'origin': np.array([-100, -100]),
    'delta_weight': 0.2,
    'steps_per_iteration': 400,
    'total_timesteps': 2000000,
    'gamma': 0.99
}

config_test = {
    'env_id': "mo-mountaincarcontinuous-v0",
    'seed': 42,
    'ref_point': np.array([-110, -110]),
    'num_envs': 4,
    'pop_size': 4,
    'warmup_iterations': 1,
    'evolutionary_iterations': 1,
    'num_weight_candidates': 50,
    'origin': np.array([-110, -110]),
    'steps_per_iteration': 100,
    'delta_weight': 0.2,
    'total_timesteps': 5000,
    'gamma': 0.995
}

# A function to load a configuration
def load_config(config_name):
    if config_name == "config_1":
        return config_1
    elif config_name == "config_test":
        return config_test
    elif config_name == "config_2":
        return config_2
    elif config_name == "config_cheetah":
        return config_cheetah
    else:
        raise ValueError(f"Unknown configuration: {config_name}")
