import pandas as pd
from run_naut import run_e_naut

# --- Nautilus option ---
USE_NAUTILUS = True

# Load the data
file_path = 'results/cheetah/norm/all_cleaned.csv'
type = "no_interactive_cheetah"

df = pd.read_csv(file_path, header=None, 
                 names=['type', 'agent_id', 'Vectorial Reward', 'weights', 'bounds'])

# Remove square brackets and split the vectorial column into x and y components
df['Vectorial Reward'] = df['Vectorial Reward'].str.strip("[]")
df[['x', 'y']] = df['Vectorial Reward'].str.split(';', expand=True).astype(float)

# Min-max normalization for x and y (store min and max for later use)
min_x, max_x = df['x'].min(), df['x'].max()
min_y, max_y = df['y'].min(), df['y'].max()
for col, min_val, max_val in zip(['x', 'y'], [min_x, min_y], [max_x, max_y]):
    if max_val > min_val:
        df[col] = (df[col] - min_val) / (max_val - min_val)
    else:
        df[col] = 0.0

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

if USE_NAUTILUS:
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

    # Normalize E-NAUTILUS output using the same min/max as the DataFrame
    def normalize_point(point, min_x, max_x, min_y, max_y):
        norm_x = (point[0] - min_x) / (max_x - min_x) if max_x > min_x else 0.0
        norm_y = (point[1] - min_y) / (max_y - min_y) if max_y > min_y else 0.0
        return [norm_x, norm_y]

    enautilus_focus_a_norm = normalize_point(enautilus_focus_a, min_x, max_x, min_y, max_y)
    enautilus_focus_b_norm = normalize_point(enautilus_focus_b, min_x, max_x, min_y, max_y)
    enautilus_even_norm = normalize_point(enautilus_even, min_x, max_x, min_y, max_y)

    # Create a row for E-NAUTILUS results, matching the DataFrame columns
    enautilus_row = {
        'type': 'e-nautilus',
        'utility_focus_a': user_utility_focus_a(*enautilus_focus_a_norm),
        'utility_focus_b': user_utility_focus_b(*enautilus_focus_b_norm),
        'utility_even': user_utility_even(*enautilus_even_norm)
    }

    # Append the row to the DataFrame
    best_values = pd.concat([best_values, pd.DataFrame([enautilus_row])], ignore_index=True)

print(best_values)
print(f"e-nautilus_selected_even_point,0,{enautilus_even[0]};{enautilus_even[1]}")
print(f"e-nautilus_selected_focus_a_point,1,{enautilus_focus_a[0]};{enautilus_focus_a[1]}")
print(f"e-nautilus_selected_focus_b_point,2,{enautilus_focus_b[0]};{enautilus_focus_b[1]}")