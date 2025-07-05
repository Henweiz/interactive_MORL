import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Filtering options ---
FILTER_BOUNDS = True
FILTER_PARETO = True
EXCLUDE_TYPE = "no_interactive_cheetah"  # Set to a type string to exclude, e.g., "no_interactive_cheetah_55_1"

# Read the data from CSV file
df = pd.read_csv('results/cheetah/norm/all_cleaned.csv', header=None, names=['type', 'agent_id', 'Vectorial Reward', 'weights', 'bounds', 'time'])

# Remove square brackets and split the vectorial column into x and y components
df['Vectorial Reward'] = df['Vectorial Reward'].str.strip("[]")  # Remove square brackets
df[['x', 'y']] = df['Vectorial Reward'].str.split(';', expand=True).astype(float)

# Split the bounds column into x and y components
df['bounds'] = df['bounds'].str.strip("[]")  # Remove square brackets
df[['bound_x', 'bound_y']] = df['bounds'].str.split(';', expand=True).astype(float)

# Exclude a specific type if set
if EXCLUDE_TYPE is not None:
    df = df[df['type'] != EXCLUDE_TYPE]

# Apply bounds filtering if enabled
if FILTER_BOUNDS:
    filtered_df = df[(df['x'] >= df['bound_x']) & (df['y'] >= df['bound_y'])]
else:
    filtered_df = df

# Pareto filtering function
def pareto_filter(df, x_col='x', y_col='y'):
    data = df[[x_col, y_col]].values
    is_pareto = np.ones(data.shape[0], dtype=bool)
    for i, point in enumerate(data):
        if is_pareto[i]:
            is_pareto[is_pareto] = np.any(data[is_pareto] > point, axis=1) | np.all(data[is_pareto] == point, axis=1)
            is_pareto[i] = True  # Keep self
    return df[is_pareto]

# Apply Pareto filtering within each type if enabled
if FILTER_PARETO:
    plot_df = pd.concat(
        [pareto_filter(group) for _, group in filtered_df.groupby('type')],
        ignore_index=True
    )
else:
    plot_df = filtered_df

# Define a color map for the different types
colors = ['blue', 'red', 'pink', 'cyan', 'green', 'yellow', 'orange', 'purple']
color_map = {type_name: colors[i % len(colors)] for i, type_name in enumerate(plot_df['type'].unique())}

# Create the plot
plt.figure(figsize=(12, 7))

# Loop through each unique type and plot
for type_name, color in color_map.items():
    subset = plot_df[plot_df['type'] == type_name]
    # Scatter plot for vectorial rewards
    plt.scatter(subset['x'], subset['y'], color=color, label=type_name, alpha=1, s=100)

# Add labels and title
plt.xlabel('X Component', fontsize=12)
plt.ylabel('Y Component', fontsize=12)
plt.title('Agent Vectorial Values', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Adjust layout and show
plt.tight_layout()
plt.savefig('filtered_agent_vectorial_values.png', dpi=300, bbox_inches='tight')
plt.show()