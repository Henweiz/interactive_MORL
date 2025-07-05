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


def run_e_naut(type='no_interactive', file_path='agent_performance.csv', artificial=True, user_utility=lambda a, b: a * 0.5 + b * 0.5):
    df = pd.read_csv(file_path, header=None, names=['type', 'agent_id', 'vectorial', 'weights', 'bounds', 'time'])

    # Convert the 'vectorial' column from strings to lists of floats
    df['vectorial'] = df['vectorial'].apply(lambda x: np.array(list(map(float, x.strip("[]").replace(" ", "").split(";")))))

    # Filter for non-interactive agents
    non_interactive = df[df['type'] == type]

    # Convert the 'vectorial' column to a 2D NumPy array
    vectorial_array = np.vstack(non_interactive['vectorial'].values)

    # Pareto filtering function
    def pareto_filter(points):
        is_pareto = np.ones(points.shape[0], dtype=bool)
        for i, point in enumerate(points):
            if is_pareto[i]:
                is_pareto[is_pareto] = np.any(points[is_pareto] > point, axis=1) | np.all(points[is_pareto] == point, axis=1)
                is_pareto[i] = True  # Keep self
        return points[is_pareto]

    # Pareto filter the points before running E_NAUTILUS
    pareto_points = pareto_filter(vectorial_array)

    # Pass the Pareto-filtered array to E_NAUTILUS
    e_nautilus = E_NAUTILUS(pareto_points, artificial=artificial, user_utility=user_utility)

    selected_solution = e_nautilus.run()
    print("Final Selected Solution:", selected_solution)
    return selected_solution

if __name__ == "__main__":
    run_e_naut(type="no_interactive_mocar", file_path="results/car/car.csv")