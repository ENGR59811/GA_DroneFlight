# The City College of New York, City University of New York
# Written by Olga Chsherbakova
# Date: September, 2023
# 
#
#
# UAV Control with Genetic Algorithm 
# Given the number of UAVs in a swarm and communication range, uniformly 
# spread the swarm such that they retain connectivity with neighbors 
# while maintaining a distance close to radius from all neighbors 

from deap import base, creator, tools, algorithms
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np
import random

n_UAV = 10 # 20 total moves for each UAV to make 
speed = 1 # constant speed of each UAV
rcom = 10 # radius of communication
n_times = 10 # number of UAVs = 20
solutions_per_pop = 10
num_generations = 50

# Give each UAV a random starting location within half rcom of 0
# we multiply by rcom/sqrt(2) so that the distance from (0,0,rcom) 
# is always less than rcom.
# UAV_positions is a 2-d matrix: [uav_no][x,y,z]

UAV_positions = np.zeros(shape= (n_times + 1, n_UAV, 3))
for x in range (0, n_UAV):
    for y in range (0, 3):
        UAV_positions[0][x][y] = (rcom/math.sqrt(2)) * random.random()

        # we will assume that we want the UAVs to begin GA at a altitude 
        # higher than rcom (this makes it less likely to crash in the ground)
        if (y%2 == 2):
            UAV_positions[0][x][y] += rcom
print("Starting Coordinates: ")
for i in range(0, n_UAV):
    print("UAV #{}:\t{}".format(i, UAV_positions[0][i]))

# Chromosomes
# The chromosome will represent the direction for the UAV to move
# Each chromosome will be represented as a vector of length 5 
# Where the most significant 2 bits will represent altitude as such:
# [0, 0, x, x, x] = Stationary 
# [0, 1, x, x, x] = Down 
# [1, 0, x, x, x] = Up
# [1, 1, x, x, x] = Maintain current altitude
# There are 4 cardinal directions and 4 ordinal directions:
# North, Northeast, East, Southeast, South, Southwest, West, Northwest
# we can encode with the least significant 3 bits of the chromosome:
#   [x, x, 0,0,0] = North   [x, x, 0,0,1] = North-East
#   [x, x, 0,1,0] = East    [x, x, 0,1,1] = South-East
#   [x, x, 1,0,0] = South   [x, x, 1,0,1] = South-West
#   [x, x, 1,1,0] = West    [x, x, 1,1,1] = North-West
#
# Since there are no restrictions on the chromosomes other than length,
# there is no need for the A and b constraint matrices.
# The fitness function for these chromosomes are at the end of the script

chromosome_length = 5

# Displacement vectors on the X-Y plane 
# Each direction and speed generate a displacement vector. 
# The magnitude of this displacement vector is always equal to speed. 
#
# Using the encodings for each direction, we can create an 2D array 
# that holds the unit displacement vectors, then multiply all the unit
# vectors by speed. The array dimensions are 
# displacement_vector[8][2] - 8 possible directions, 2 for x displacement
# and y displacement.
#
# for example, using the three least significant bits:
# (index) [unit displacement vector] 
# 0       [0, 1]                       - north      (xx000 is binary 0)
# 1       [1/sqrt(2), 1/sqrt(2)]       - northeast  (xx001 is binary 1)
# 2       [1, 0]                       - east       (xx010 is binary 2)
# 3       [1/sqrt(2), -1/sqrt(2)]      - southeast  (xx011 is binary 3)
# 4       [0, -1]                      - south      (xx100 is binary 4)
# 5       [-1/sqrt(2)), -1/sqrt(2)]    - southwest  (xx101 is binary 5) 
# 6       [-1, 0]                      - west       (xx110 is binary 6)
# 7       [-1/sqrt(2)), 1/sqrt(2)]     - northwest  (xx110 is binary 7) 

# chromosome : b2b1b0 -> convert to decimal 0-7 and use as an index:
# [N, NE, E, SE, S, SW, W, NW]
# chromosome 101 -> 5 -> SW

displacement_vectors = np.array([[0, 1], [1/math.sqrt(2), 1/math.sqrt(2)], 
                        [1, 0], [1/math.sqrt(2), -1/math.sqrt(2)], 
                        [0, -1], [-1/math.sqrt(2), -1/math.sqrt(2)], 
                        [-1, 0], [-1/math.sqrt(2), 1/math.sqrt(2)]])
displacement_vectors = displacement_vectors * speed

# The calc_distance calculates distance between two points in 3D space. 
def calc_distance(old_coordinates, new_coordinates):
    return math.sqrt((new_coordinates[0] - old_coordinates[0])**2 + \
                     (new_coordinates[1] - old_coordinates[1])**2 + \
                         (new_coordinates[2] - old_coordinates[2])**2)

# The decode function appears to be related to decoding a chromosome into a 
# displacement vector. It takes a chromosome as input, which be represented 
# as a binary sequence, and decodes it into a 3D displacement vector.
def decode(chromosome):
    # initializes a NumPy array called displacement to store the 
    # resulting displacement vector. It initializes it with three zeros, 
    # representing the (x, y, z) components of the displacement.
    displacement = np.zeros((3,), dtype=int)
    # If chromosome[0] is 0 and chromosome[1] is 0, it means there's no change
    # in the z-direction, so displacement[2] remains 0.
    if (chromosome[0] == 0 and chromosome[1] == 0):
        return displacement
    #If chromosome[0] is 0 and chromosome[1] is 1, it sets displacement[2] 
    # to 1 * speed, implying a positive change in the z-direction.
    if (chromosome[0] == 0 and chromosome[1] == 1):
        displacement[2] = 1 * speed
    # If chromosome[0] is 1 and chromosome[1] is 0, it sets displacement[2] 
    # to -1 * speed, implying a negative change in the z-direction.
    if (chromosome[0] == 1 and chromosome[1] == 0):
        displacement[2] = -1 * speed
    #If chromosome[0] is 1 and chromosome[1] is 1, it sets displacement[2] 
    # to 0, which means that there is no change in the z-direction.
    if (chromosome[0] == 1 and chromosome[1] == 1):
        displacement[2] = 0
    
    # calculates an index based on the values of three elements in the 
    # chromosome array to select a displacement vector from some predefined 
    # array based on the values of these elements.
    LSB_index = 1 * chromosome[4] + 2 * chromosome[3] + 4 * chromosome[2]
    # sets the displacement[0] and displacement[1] components of the 
    # displacement vector using values from array called displacement_vectors. 
    # The specific values used are determined by the LSB_index calculated 
    # in the previous step.
    displacement[0], displacement[1] = \
        displacement_vectors[LSB_index][0], displacement_vectors[LSB_index][1]
    return displacement

# This function finds neighboring UAVs for a given UAV at a specific time.
def get_neighbors(time, UAV_index):
    neighbors = []
    for i in range(0, n_UAV):
        UAV = UAV_positions[time][i]
        # If the distance between the current UAV and the target UAV is less 
        # than or equal to rcom, indicating that they are within communication 
        # range, the current UAV is considered a neighbor, and its position 
        # is appended to the neighbors list.
        if (UAV_index != i and calc_distance(UAV, 
                UAV_positions[time][UAV_index]) <= rcom):
            neighbors.append(UAV)
    return neighbors  

# Calculate fitness score based on its displacement, position, and neighbors.
def fit_func(individual, current_UAV_coordinates, neighbors):
    fitness_score = 0
    displacement = decode(individual)
    new_position = [current_UAV_coordinates[0] + displacement[0], 
                    current_UAV_coordinates[1] + displacement[1], 
                    current_UAV_coordinates[2] + displacement[2]] 
    if (new_position[2] < 0 ):
        return [np.inf]
    for neighbor in neighbors:
        distance = calc_distance(neighbor, new_position)
        fitness_score += (rcom - distance)
    return [fitness_score]

def animate(UAV_positions):
    # Create figure object.
    fig = plt.figure()
    # Create 3D axis object using add_subplot().
    ax = fig.add_subplot(111, projection='3d')
    u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
    unit_sph_x = np.cos(u)*np.sin(v)
    unit_sph_y = np.sin(u)*np.sin(v)
    unit_sph_z = np.cos(v)
    fig.show()
    for i in range(0, n_times + 1):
        ax.clear()
        for j in range (0, n_UAV):
            sph_x = (rcom*unit_sph_x + UAV_positions[i][j][0])
            sph_y = (rcom*unit_sph_y + UAV_positions[i][j][1])
            sph_z = (rcom*unit_sph_z + UAV_positions[i][j][2])
            ax.plot_surface(sph_x,sph_y,sph_z, color='w', alpha = 0.05, 
                           edgecolor = 'white')
        ax.scatter(UAV_positions[i, :, 0], UAV_positions[i, :, 1], 
                           UAV_positions[i, :, 2], s = 30, color = 'r' )
        fig.canvas.draw()
        plt.pause(0.25)
        fig.canvas.flush_events()
    plt.show()

if hasattr(creator, "Fitness"):
   del creator.Fitness
if hasattr(creator, "Individual"):
   del creator.Individual

INDIVIDUAL_SIZE = chromosome_length
creator.create("Fitness", base.Fitness, weights=(-1.0, ))
creator.create("Individual", list, fitness=creator.Fitness)
toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual",  tools.initRepeat,  creator.Individual,
             toolbox.attr_bool, n=INDIVIDUAL_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)
toolbox.register("select", tools.selTournament, tournsize=5)

def main():
    # CXPB  is the probability with which two individuals
    #       are crossed
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.2
    for i in range(0, n_times):
        print("\nTIME #{}\n".format(i))
        for j in range(0, n_UAV):
            print("\nRUNNING GA FOR UAV #{}\n".format(j))
            UAV = UAV_positions[i][j]
            print("CURRENT COORDINATES:\t{}\n".format(UAV))
            neighbors = np.array(get_neighbors(i,j))
            # print("NEIGHBORS FOR UAV #{}\n{}".format(i, neighbors))
            toolbox.register("evaluate", fit_func, current_UAV_coordinates = 
                             UAV, neighbors = neighbors )
            pop = toolbox.population(solutions_per_pop)
            hof = tools.HallOfFame(1)
            pop, log = algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, 
                            num_generations, verbose= False, halloffame=hof)
            new_coordinates = (UAV + decode(hof[0]))
            print("NEW COORDINATES:\t{}\n".format(new_coordinates))  
            UAV_positions[i+1][j] = new_coordinates
              
    animate(UAV_positions)

main()
# animate()