import matplotlib.pyplot as plt
import numpy as np

def interactive_plot(pareto_points, agents=None):
     # Interactive plot
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(pareto_points[:, 0], pareto_points[:, 1], color='blue', picker=True)  # Set all points to blue
        ax.set_xlabel("Objective 1")
        ax.set_ylabel("Objective 2")
        ax.set_title("Interactive Pareto Front Selection")
        selected_point_marker = None  # Marker for the selected point

        selected_agent = None
        selected_evaluation = None

        def on_click(event):
            nonlocal selected_agent, selected_evaluation, selected_point_marker
            if event.inaxes != ax:
                return

            # Get the clicked coordinates
            clicked_x, clicked_y = event.xdata, event.ydata
            print(f"Clicked at: ({clicked_x}, {clicked_y})")

            # Find the closest point in the Pareto front
            distances = np.linalg.norm(pareto_points - np.array([clicked_x, clicked_y]), axis=1)
            closest_index = np.argmin(distances)
            selected_evaluation = pareto_points[closest_index]

            # Retrieve the selected agent and evaluation
            if agents is not None:
                selected_agent = agents[closest_index] if closest_index < len(agents) else None

            if selected_agent is not None:
                print(f"Selected Agent: {selected_agent.id}")
            print(f"Closest Point: {selected_evaluation}")

            # Update the plot to highlight the selected point
            if selected_point_marker:
                selected_point_marker.remove()  # Remove the previous marker
            selected_point_marker = ax.scatter(
                selected_evaluation[0], selected_evaluation[1], color='red', s=100, label="Selected Point"
            )
            ax.legend()
            plt.draw()  # Update the plot to show the highlighted point

        # Connect the click event to the on_click function
        fig.canvas.mpl_connect('button_press_event', on_click)

        plt.show()
        return selected_agent, selected_evaluation