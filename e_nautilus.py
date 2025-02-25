import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt

class E_NAUTILUS:
    def __init__(self, nd_solutions):
        self.nd_solutions = nd_solutions
        self.nadir = np.min(nd_solutions, axis=0)
        self.ideal = np.max(nd_solutions, axis=0)
        self.selected_points = []  # Store all selected points
        self.history = []  # Store full history of selection

    #TODO: Add failsafe for when the user inputs weird stuff
    def initial_pref(self):
        print(f"Nadir vector: {self.nadir}")
        print(f"Ideal vector: {self.ideal}")
        print("How many iterations would you like to carry out?")
        iterations = int(input())
        print("How many points would you like to investigate at each iteration?")
        points = int(input())
        return iterations, points
        

    def run(self):
        iterations, n_points = self.initial_pref()
        z = self.nadir
        p = self.nd_solutions
        id = 0

        for h in range(iterations, 0, -1):
            if len(p) <= n_points:
                print("Number of points is less than the number of clusters. Using all points.")
                h = 1
                break
                
            clusters = self.cluster_front(n_points, p)
            z = self.generate_points(z, h, clusters, n_points)

            self.history.append(z.copy())  # Store progression
            

            for i, point in enumerate(z):
                print(f"#{i} \n point: {point}")
            id = int(input("Enter the id of the point you would like to choose as the preferred point: "))
            z = z[id]
            self.selected_points.append(z)

            p = self.bound_points(p, z)
            print(f"Closeness to reference point: {self.closeness(z, clusters[id])}")
            self.plot_progression()  # Update visualization
        
        if len(p) < 1:
            print(f"Return the selected point: {z}")
            return self.post_process(z)
        
        for i, point in enumerate(p):
            print(f"#{i} \n point: {point}")
        id = int(input("Enter the id of the point you would like to choose as the FINAL preferred point: "))

        self.selected_points.append(p[id])
        self.plot_progression()
        return p[id]
            
    def post_process(self, z):
        p = np.array(self.nd_solutions)  # Convert list to NumPy array for efficiency
        z = np.array(z)  # Ensure z is also a NumPy array

        # Compute Euclidean distances
        distances = np.linalg.norm(p - z, axis=1)

        # Find the index of the closest point
        closest_index = np.argmin(distances)
        
        # Retrieve the closest point
        closest_point = p[closest_index]

        print(f"Post-processing... Closest point to selected preferred solution: {closest_point}")

        return closest_point
    
    def closeness(self, z_h, z_ref):
        # Compute Euclidean distances
        numerator = np.linalg.norm(z_h - self.nadir)  # Distance from nadir to intermediate point
        denominator = np.linalg.norm(z_ref - self.nadir)  # Distance from nadir to Pareto reference
        
        if denominator == 0:
            raise ValueError("Denominator is zero! Ensure z_ref is not the same as z_nad.")
        
        closeness = (numerator / denominator) * 100  # Convert to percentage
        return closeness
    
    def bound_points(self, points, bounds):
        bounded_points = [point for point in points if np.all(point >= bounds)]
        return np.array(bounded_points)  

    
    def generate_points(self, prev_z, h, clusters, n_points):
        points = []
        for i in range(n_points):
            z = self.compute_intermediate_point(prev_z, clusters[i], h)
            points.append(z)
        return points

    def compute_intermediate_point(self, prev_point, reference_point, iteration):
        """
        Computes the intermediate point in the E-NAUTILUS method.
        
        Parameters:
        - prev_z (numpy array): The previous intermediate point (z^{h-1}).
        - reference_z (numpy array): The reference Pareto-optimal solution (z̄).
        - iteration (int): The current iteration number (it^h).
        
        Returns:
        - numpy array: The new intermediate point (z^h).
        """
        if iteration <= 0:
            raise ValueError("Iteration count must be positive.")
        
        factor = (iteration - 1) / iteration
        new_z = factor * prev_point + ((1 / iteration) * reference_point)

        # Check for NaN values in the result
        if np.any(np.isnan(new_z)):
            raise ValueError(f"Computed intermediate point is NaN. prev_point: {prev_point}, reference_point: {reference_point}, iteration: {iteration}")
        return new_z


    #TODO: Check for NAN values in the result
    def cluster_front(self, n_points, p):
        if len(p) < n_points:
            print("Number of points is less than the number of clusters. Using all points.")
            return p

        Z = linkage(p, method='average')
        clusters = fcluster(Z, t=n_points, criterion='maxclust')

        clustered_points = []
        for i in range(1, n_points + 1):
            indices = np.where(clusters == i)[0]  
            cluster = p[indices]  # Extract all points in cluster
            
            # Compute cluster center (centroid)
            centroid = np.mean(cluster, axis=0)
        
            clustered_points.append(centroid)  

        return np.array(clustered_points)
    
    def plot_progression(self):
        """
        Plots the Pareto front, intermediate points, and selected points.
        Adds vertical and horizontal lines only for the most recent selected point.
        """
        pareto_x, pareto_y = zip(*self.nd_solutions)
        selected_x, selected_y = zip(*self.selected_points)
        
        plt.figure(figsize=(8, 6))
        
        # Plot Pareto front
        plt.scatter(pareto_x, pareto_y, color='green', label="Pareto Front (All Points)")
        
        # Plot intermediate selections
        for step, step_points in enumerate(self.history):
            step_x, step_y = zip(*step_points)
            label = "Previous Iteration(s)" if step == 0 else None
            plt.scatter(step_x, step_y, color='gray', alpha=0.3, label=label)

        # Plot user-selected points
        plt.scatter(selected_x[:-1], selected_y[:-1], color='blue', marker='x', s=100, label="Intermediate Selections")
        
        # Plot final user choice
        plt.scatter(selected_x[-1], selected_y[-1], color='red', marker='*', s=200, label="Final Choice")
        
        # Draw path of selected points
        plt.plot(selected_x, selected_y, linestyle='--', color='gray', label="Selection Path")
        
        # Draw vertical and horizontal lines only for the most recent selection
        if self.selected_points:
            latest_x, latest_y = self.selected_points[-1]
            plt.axvline(x=latest_x, color='blue', linestyle='--', alpha=0.5)  # Vertical line for latest point
            plt.axhline(y=latest_y, color='blue', linestyle='--', alpha=0.5)  # Horizontal line for latest point

        plt.xlabel("Objective 1")
        plt.ylabel("Objective 2")
        plt.title("E-NAUTILUS Progression with History and Boundaries")
        plt.legend()
        plt.grid(True)
        plt.show()

