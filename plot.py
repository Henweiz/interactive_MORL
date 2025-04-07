import pandas as pd
import matplotlib.pyplot as plt

# Read the data from CSV file
df = pd.read_csv('agent_performance.csv', header=None, names=['type', 'agent_id', 'vectorial', 'weights'])

# Remove square brackets and split the vectorial column into x and y components
df['vectorial'] = df['vectorial'].str.strip("[]")  # Remove square brackets
df[['x', 'y']] = df['vectorial'].str.split(';', expand=True).astype(float)

# Create the plot
plt.figure(figsize=(12, 7))

# Plot interactive points (blue)
interactive = df[df['type'] == 'interactive']
plt.scatter(interactive['x'], interactive['y'], 
           color='blue', label='Interactive 1', alpha=0.7, s=100)

# Plot non-interactive points (red)
non_interactive = df[df['type'] == 'no_interactive']
plt.scatter(non_interactive['x'], non_interactive['y'], 
           color='red', label='Non-Interactive', alpha=0.7, s=100)

non_interactive_2 = df[df['type'] == 'no_interactive_2']
plt.scatter(non_interactive_2['x'], non_interactive_2['y'], 
           color='pink', label='Non-Interactive-Low', alpha=0.7, s=100)

interactive_2 = df[df['type'] == 'interactive_2']
plt.scatter(interactive_2['x'], interactive_2['y'], 
           color='cyan', label='Interactive-Low', alpha=0.7, s=100)

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