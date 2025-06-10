import pandas as pd
from run_naut import run_e_naut

# Load the data
file_path = 'results/car/car.csv'
type = "no_interactive_mocar"

df = pd.read_csv(file_path, header=None, 
                 names=['type', 'agent_id', 'Vectorial Reward', 'weights', 'bounds'])

# Remove square brackets and split the vectorial column into x and y components
df['Vectorial Reward'] = df['Vectorial Reward'].str.strip("[]")
df[['x', 'y']] = df['Vectorial Reward'].str.split(';', expand=True).astype(float)

# Define user utility functions
def user_utility_focus_a(a, b):
    return a * 0.7 + b * 0.3

def user_utility_focus_b(a, b):
    return a * 0.3 + b * 0.7

def user_utility_even(a, b):
    return a * 0.5 + b * 0.5

def only_a(a, b):
    return a
def only_b(a, b):
    return b


# Add utility values to the DataFrame
df['utility_focus_a'] = df.apply(lambda row: user_utility_focus_a(row['x'], row['y']), axis=1)
df['utility_focus_b'] = df.apply(lambda row: user_utility_focus_b(row['x'], row['y']), axis=1)
df['utility_even'] = df.apply(lambda row: user_utility_even(row['x'], row['y']), axis=1)

# Find the best value for each type and utility function
best_values = df.groupby('type').agg({
    'utility_focus_a': 'max',
    'utility_focus_b': 'max',
    'utility_even': 'max',
}).reset_index()

# Compute E-NAUTILUS results for each utility function
enautilus_focus_a = run_e_naut(
    file_path=file_path,
    type=type,
    artificial=True,
    user_utility=user_utility_focus_a
)
enautilus_focus_b = run_e_naut(
    file_path=file_path,
    type=type,
    artificial=True,
    user_utility=user_utility_focus_b
)
enautilus_even = run_e_naut(
    file_path=file_path,
    type=type,
    artificial=True,
    user_utility=user_utility_even
)

# Create a row for E-NAUTILUS results, matching the DataFrame columns
enautilus_row = {
    'type': 'e-nautilus',
    'utility_focus_a': user_utility_focus_a(*enautilus_focus_a),
    'utility_focus_b': user_utility_focus_b(*enautilus_focus_b),
    'utility_even': user_utility_even(*enautilus_even)
}

# Append the row to the DataFrame
best_values = pd.concat([best_values, pd.DataFrame([enautilus_row])], ignore_index=True)

print(best_values)