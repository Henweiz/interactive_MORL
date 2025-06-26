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
    df = pd.read_csv(file_path, header=None, names=['type', 'agent_id', 'vectorial', 'weights', 'bounds'])

    # Debug: Print the first few rows of the DataFrame to inspect the 'vectorial' column
    print("Original DataFrame:")
    print(df.head())

    # Convert the 'vectorial' column from strings to lists of floats
    df['vectorial'] = df['vectorial'].apply(lambda x: np.array(list(map(float, x.strip("[]").replace(" ", "").split(";")))))

    # Debug: Print the processed 'vectorial' column to ensure proper conversion
    print("Processed 'vectorial' column:")
    print(df['vectorial'].head())

    # Filter for non-interactive agents
    non_interactive = df[df['type'] == type]
    print("Filtered non-interactive agents:")
    print(non_interactive.head())

    # Convert the 'vectorial' column to a 2D NumPy array
    vectorial_array = np.vstack(non_interactive['vectorial'].values)

    # Debug: Print the resulting 2D NumPy array
    print("2D NumPy array:")
    print(vectorial_array)

    # Pass the 2D array to E_NAUTILUS
    e_nautilus = E_NAUTILUS(vectorial_array, artificial=artificial, user_utility=user_utility)

    selected_solution = e_nautilus.run()
    print("Final Selected Solution:", selected_solution)
    return selected_solution

if __name__ == "__main__":
    run_e_naut(type="no_interactive_mocar", file_path="results/car/car.csv")