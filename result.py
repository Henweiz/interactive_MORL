import pandas as pd

# Load the data
df = pd.read_csv('results/cheetah/all.csv', header=None, 
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
# Add utility functions for only a and only b
df['only_a'] = df.apply(lambda row: only_a(row['x'], row['y']), axis=1)
df['only_b'] = df.apply(lambda row: only_b(row['x'], row['y']), axis=1)

# Add utility values to the DataFrame
df['utility_focus_a'] = df.apply(lambda row: user_utility_focus_a(row['x'], row['y']), axis=1)
df['utility_focus_b'] = df.apply(lambda row: user_utility_focus_b(row['x'], row['y']), axis=1)
df['utility_even'] = df.apply(lambda row: user_utility_even(row['x'], row['y']), axis=1)

# Find the best value for each type and utility function
best_values = df.groupby('type').agg({
    'utility_focus_a': 'max',
    'utility_focus_b': 'max',
    'utility_even': 'max',
    'only_a': 'max',
    'only_b': 'max'
}).reset_index()

# Print the results
print(best_values)