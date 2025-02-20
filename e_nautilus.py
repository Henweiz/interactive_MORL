import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist

class E_NAUTILUS:
    def __init__(self, nd_solutions):
        self.nd_solutions = nd_solutions
        self.nadir = np.min(nd_solutions, axis=0)
        self.ideal = np.max(nd_solutions, axis=0)

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
        z = [self.nadir for _ in range(n_points)]
        p = self.nd_solutions
        for h in range(iterations, 0, -1):

            clusters = self.cluster_front(n_points, p)
            z = self.generate_points(z, h, clusters, n_points)
            for i, point in enumerate(z):
                print(f"#{i} \n point: {point}")
            id = int(input("Enter the id of the point you would like to choose as the preferred point: "))
            #pref_point = p[id]
            p = self.bound_points(p, z[id])
            print(f"Closeness to reference point: {self.closeness(z[id], clusters[id])}")
        
        for i, point in enumerate(p):
            print(f"#{i} \n point: {point}")
        id = int(input("Enter the id of the point you would like to choose as the FINAL preferred point: "))
        return p[id]
            
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
            z = self.compute_intermediate_point(prev_z[i], clusters[i], h)
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
        new_z = factor * prev_point + (1 / iteration) * reference_point
        return new_z



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
            
            # Find the closest actual point to the centroid
            closest_idx = np.argmin(cdist([centroid], cluster))  
            cluster_representative = cluster[closest_idx]
            
            clustered_points.append(cluster_representative)  

        return np.array(clustered_points)


