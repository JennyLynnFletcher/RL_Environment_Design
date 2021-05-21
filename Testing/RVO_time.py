import timeit

setup = '''
from Simulator import Simulation

s = Simulation()

s.add_robot_on_matrix(2,2, 3,0, 200, 6)
s.add_robot_on_matrix(3,3, 1,2, 200, 6)


s.add_obstacle_matrix( [[1, 0, 0, 0, 0],
                        [0, 1, 0, 1, 0],
                        [1, 0, 0, 1, 1],
                        [0, 0, 0, 0, 0],
                        [0, 1, 1, 0, 0]])'''
                        
code = '''
for i in range(500):
    if not all(s.arrived):
        s.step_simulation()
    else:
        break
'''

print("Example 5x5 environment with 2 robots")
print (timeit.timeit(setup = setup,
                     stmt = code,
                     number = 100)) 

setup = '''
from Simulator import Simulation

s = Simulation()

s.add_robot_on_matrix(0,0, 5,5, 200, 6)
s.add_robot_on_matrix(0,5, 5,0, 200, 6)
s.add_robot_on_matrix(5,0, 0,5, 200, 6)
s.add_robot_on_matrix(5,5, 0,0, 200, 6)

'''

print("No obstacles 4 robots in square swapping places")
print (timeit.timeit(setup = setup,
                     stmt = code,
                     number = 100)) 


setup = '''
from Simulator import Simulation

s = Simulation()

s.add_robot_on_matrix(0,4, 4,4, 200, 6)
s.add_robot_on_matrix(10,10, 4,5, 200, 6)


s.add_obstacle_matrix( [[0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],                        
                        ])'''

print("10x10 high obstacle density")
print (timeit.timeit(setup = setup,
                     stmt = code,
                     number = 100)) 
