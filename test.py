import numpy as np
from e_nautilus import E_NAUTILUS

def generate_pareto_front(n_points=10, min_value=-10, max_value=10):
    """
    Generates a 2D Pareto front with maximization.
    
    Parameters:
    - n_points (int): Number of Pareto-efficient solutions.
    - max_value (float): Upper bound for objective values.
    
    Returns:
    - numpy array of shape (n_points, 2): Pareto-optimal solutions.
    """
    f1 = np.linspace(min_value, max_value, n_points)  # Evenly spaced values for Obj 1
    f2 = -((f1 - min_value) ** 2) / abs(min_value)  # Convex function for Pareto front
    pareto_front = np.column_stack((f1, f2))  # Stack to form (n_points, 2)

    return pareto_front



if __name__ == "__main__":

    # Generate 10 random non-dominated solutions with 2 objectives
    np.random.seed(42)  # For reproducibility
    test_nd_solutions = generate_pareto_front(n_points=30, min_value=-20, max_value=0)

    # Print the generated test data
    print("Test Non-Dominated Solutions:")
    print(test_nd_solutions.tolist())
    # Initialize the E-NAUTILUS class with test data
    e_nautilus = E_NAUTILUS(test_nd_solutions)

    # Run the interactive method
    selected_solution = e_nautilus.run()

    print("Final Selected Solution:", selected_solution)
