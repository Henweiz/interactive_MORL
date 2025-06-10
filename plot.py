import pandas as pd
import matplotlib.pyplot as plt

# Read the data from CSV file
df = pd.read_csv('results/cheetah/42.csv', header=None, names=['type', 'agent_id', 'Vectorial Reward', 'weights', 'bounds'])

# Remove square brackets and split the vectorial column into x and y components
df['Vectorial Reward'] = df['Vectorial Reward'].str.strip("[]")  # Remove square brackets
df[['x', 'y']] = df['Vectorial Reward'].str.split(';', expand=True).astype(float)

# Split the bounds column into x and y components
df['bounds'] = df['bounds'].str.strip("[]")  # Remove square brackets
df[['bound_x', 'bound_y']] = df['bounds'].str.split(';', expand=True).astype(float)

# Filter the DataFrame to include only rows where rewards are better than bounds
filtered_df = df[(df['x'] >= df['bound_x']) & (df['y'] >= df['bound_y'])]
#filtered_df = df

# Define a color map for the different types
colors = ['blue', 'red', 'pink', 'cyan', 'green', 'yellow', 'orange', 'purple']
color_map = {type_name: colors[i % len(colors)] for i, type_name in enumerate(filtered_df['type'].unique())}

# Create the plot
plt.figure(figsize=(12, 7))

# Loop through each unique type and plot
for type_name, color in color_map.items():
    subset = filtered_df[filtered_df['type'] == type_name]
    # Scatter plot for vectorial rewards
    plt.scatter(subset['x'], subset['y'], color=color, label=type_name, alpha=0.7, s=100)

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