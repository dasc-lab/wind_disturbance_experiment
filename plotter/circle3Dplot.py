import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_trajectory(num_points, radius, height):
    waypoints = []
    for i in range(num_points):
        theta = 2 * np.pi * i / num_points
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        waypoints.append((x, y, height))
    return waypoints





def main(args=None):

    waypoints = generate_trajectory(50, 1.5, 1.5)
    second_waypoint = waypoints
    second_waypoint = np.array(second_waypoint)
    second_waypoint[:,2] = second_waypoint[:,2] - 0.2
    print(waypoints)
    x = np.arange(1,len(waypoints)+1)
    
    waypoints = np.array(waypoints)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax = fig.add_subplot(211, projection='3d')
    #ax2 = fig.add_subplot(212, projection)
    x = waypoints[:,0]
    y = waypoints[:,1]
    z = waypoints[:,2]
    # Scatter plot
    ax.scatter(x, y, z, c= 'b')
   
    x = second_waypoint[:,0]
    y = second_waypoint[:,1]
    z = second_waypoint[:,2]
    # Scatter plot
    ax.scatter(x, y, z, c = 'r')
    
    # plt.scatter(waypoints[:,0], waypoints[:,1], waypoints[:,2], color='blue', label='Data Points', projection='3d')


    # Add labels and title
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.zlabel('Z-axis')
    # plt.title('Scatter Plot Example')

    # Add legend
    plt.legend()

    # Show plot
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()

