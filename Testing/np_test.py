import numpy as np

GRID_SIZE = 5
    
NUM_ROBOTS = 1  
    
MAX_STEPS = 1000
    
state = np.zeros(((2 * GRID_SIZE) + 2, (2 * GRID_SIZE) + 2, NUM_ROBOTS + 1), dtype=int)
        
xy = np.mgrid[:GRID_SIZE,:GRID_SIZE].reshape(2, -1).T
start_positions = xy.take(np.random.choice(xy.shape[0], NUM_ROBOTS, replace=False), axis=0)
xy = np.mgrid[:GRID_SIZE,:GRID_SIZE].reshape(2, -1).T
end_positions = xy.take(np.random.choice(xy.shape[0], NUM_ROBOTS, replace=False), axis=0)  

print(start_positions)
print(end_positions)

for i in range(NUM_ROBOTS):  
    state[start_positions[i][1] + GRID_SIZE][start_positions[i][0] + GRID_SIZE][i] = 1
    state[end_positions[i][1] + GRID_SIZE][end_positions[i][0] + GRID_SIZE][i] = 2 

state[GRID_SIZE,GRID_SIZE,NUM_ROBOTS] = 1

offset = [0,0]

new_state = np.zeros(((2 * GRID_SIZE) + 2, (2 * GRID_SIZE) + 2, NUM_ROBOTS + 1), dtype=int)
new_state[GRID_SIZE - offset[0]: 2*GRID_SIZE - offset[0], GRID_SIZE - offset[1] - 1 : 2*GRID_SIZE - offset[1] - 1, :] = state[GRID_SIZE - offset[0]: 2*GRID_SIZE - offset[0], GRID_SIZE - offset[1] : 2*GRID_SIZE - offset[1], :]
print(new_state[:,:,1])


print(state[GRID_SIZE - offset[0] : 2*GRID_SIZE - offset[0],GRID_SIZE - offset[1] : 2*GRID_SIZE - offset[1],NUM_ROBOTS])
offset = [0,1]
print("\n")
print(new_state[GRID_SIZE - offset[0] : 2*GRID_SIZE - offset[0],GRID_SIZE - offset[1] : 2*GRID_SIZE - offset[1],NUM_ROBOTS])
print("\n")
print("\n")
state = new_state

state[GRID_SIZE,GRID_SIZE,NUM_ROBOTS] = 1

new_state = np.zeros(((2 * GRID_SIZE) + 2, (2 * GRID_SIZE) + 2, NUM_ROBOTS + 1), dtype=int)
new_state[GRID_SIZE - offset[0] - 1 : 2*GRID_SIZE - offset[0] - 1, GRID_SIZE - offset[1] : 2*GRID_SIZE - offset[1], :] = state[GRID_SIZE - offset[0]: 2*GRID_SIZE - offset[0], GRID_SIZE - offset[1] : 2*GRID_SIZE - offset[1], :]
print(new_state[:,:,1])


print(state[GRID_SIZE - offset[0] : 2*GRID_SIZE - offset[0],GRID_SIZE - offset[1] : 2*GRID_SIZE - offset[1],NUM_ROBOTS])
offset = [1,1]
print("\n")
print(new_state[GRID_SIZE - offset[0] : 2*GRID_SIZE - offset[0],GRID_SIZE - offset[1] : 2*GRID_SIZE - offset[1],NUM_ROBOTS])
print("\n")
print("\n")
state = new_state

state[GRID_SIZE,GRID_SIZE,NUM_ROBOTS] = 1

new_state = np.zeros(((2 * GRID_SIZE) + 2, (2 * GRID_SIZE) + 2, NUM_ROBOTS + 1), dtype=int)
new_state[GRID_SIZE - offset[0] - 1 : 2*GRID_SIZE - offset[0] - 1, GRID_SIZE - offset[1] : 2*GRID_SIZE - offset[1], :] = state[GRID_SIZE - offset[0]: 2*GRID_SIZE - offset[0], GRID_SIZE - offset[1] : 2*GRID_SIZE - offset[1], :]
print(new_state[:,:,1])


print(state[GRID_SIZE - offset[0] : 2*GRID_SIZE - offset[0],GRID_SIZE - offset[1] : 2*GRID_SIZE - offset[1],NUM_ROBOTS])
    
