import pandas as pd
import matplotlib.pyplot as plt

# Read the data from CSV file
df = pd.read_csv('agent_performance.csv', header=None, names=['type', 'agent_id', 'Vectorial Reward', 'weights'])

# Remove square brackets and split the vectorial column into x and y components
df['Vectorial Reward'] = df['Vectorial Reward'].str.strip("[]")  # Remove square brackets
df[['x', 'y']] = df['Vectorial Reward'].str.split(';', expand=True).astype(float)

# Define a color map for the different types
colors = ['blue', 'red', 'pink', 'cyan', 'green', 'yellow', 'orange', 'purple']
color_map = {type_name: colors[i % len(colors)] for i, type_name in enumerate(df['type'].unique())}

# Create the plot
plt.figure(figsize=(12, 7))

# Loop through each unique type and plot
for type_name, color in color_map.items():
    subset = df[df['type'] == type_name]
    plt.scatter(subset['x'], subset['y'], color=color, label=type_name, alpha=0.7, s=100)

# Add labels and title
plt.xlabel('X Component', fontsize=12)
plt.ylabel('Y Component', fontsize=12)
plt.title('Agent Vectorial Values Comparison', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Adjust layout and show
plt.tight_layout()
plt.savefig('agent_vectorial_comparison.png', dpi=300, bbox_inches='tight')
plt.show()