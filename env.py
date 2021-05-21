import numpy as np
import gym
from gym.utils import seeding
from gym import spaces
import math
import json
import pickle

import gym, ray
from ray.rllib.agents import ppo

import Simulator

def dist(pos0, pos1):
    return math.sqrt((pos0[0]-pos1[0])**2 + (pos0[1]-pos1[1])**2)

class ObstaclesEnv(gym.Env):
    
    GRID_SIZE = 5
    
    NUM_ROBOTS = 2  
    
    MAX_STEPS = 500
    
    NUM_OBSTACLES = 8
    
    ROBOT_VEL = 500

    metadata = {
        "render.modes": ["human"]
        }
    
    USE_FIXED_STARTS = True


    def __init__ (self, env_config):
        
        self.action_space = spaces.Discrete(9)
        #move I cell in any direction or stay in the same location

        self.observation_space = spaces.Box(0,self.NUM_OBSTACLES,shape=((2 * self.GRID_SIZE) + 2, (2 * self.GRID_SIZE) + 2, self.NUM_ROBOTS + 1), dtype=np.uint8)
        #grid for each robot showing start and end location, plus grid showing current obstacle layout
        
        self.offset = [0,0]
        #offset to move grid cell into centre of observation space
        
        self.iterations = 0
        #iterations of moving each obstacle in an episode
        
        self.obstacle_count = 0
    
        self.obstacles_on_robot_start = 0
        
        self.obstacles_on_robot_end = 0
        
        self.obstacles_off_robot_start = 0
        
        self.obstacles_off_robot_end = 0     
        
        self.perc_dist = []
        
        self.perc_speed = []
        
        self.obstacle_grid = np.zeros((self.GRID_SIZE, 2 * self.GRID_SIZE), dtype=int)
        
        self.starts = []
        
        self.grid_rng = np.random.RandomState(0)

        self.seed()

        self.reset()
    

    def reset (self):
        
        self.obstacle_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.state = np.zeros(((2 * self.GRID_SIZE) + 2, (2 * self.GRID_SIZE) + 2, self.NUM_ROBOTS + 1), dtype=int)
        self.robot_positions = np.zeros((self.GRID_SIZE, self.GRID_SIZE, self.NUM_ROBOTS), dtype=int)
        
        xy = np.mgrid[:self.GRID_SIZE,:self.GRID_SIZE].reshape(2, -1).T
        coordinates = xy.take(self.grid_rng.choice(xy.shape[0], 2 * self.NUM_ROBOTS + self.NUM_OBSTACLES, replace=False), axis=0)
        self.start_positions = coordinates[0:self.NUM_ROBOTS]
        self.end_positions = coordinates[self.NUM_ROBOTS:2*self.NUM_ROBOTS]  
        self.obstacle_positions = coordinates[2*self.NUM_ROBOTS:] 
        #print(coordinates)
        
        for i in range(self.NUM_OBSTACLES):
            self.obstacle_grid[self.obstacle_positions[i][1]][self.obstacle_positions[i][0]] = 1
            
        #print(self.obstacle_grid)
        
        for i in range(self.NUM_ROBOTS):  
            self.robot_positions[self.start_positions[i][1]][self.start_positions[i][0]][i] = 1
            self.robot_positions[self.end_positions[i][1]][self.end_positions[i][0]][i] = 2 

        self.reward = 0
        self.done = False
        self.info = {}
        self.offset = self.obstacle_positions[0]
        self.iterations = 0
        self.obstacle_num = 0
        
        self.state[self.GRID_SIZE - self.offset[1] + 1 : 2*self.GRID_SIZE - self.offset[1] + 1, self.GRID_SIZE - self.offset[0] + 1 : 2*self.GRID_SIZE - self.offset[0] + 1, self.NUM_ROBOTS] = self.obstacle_grid
        self.state[self.GRID_SIZE - self.offset[1] + 1 : 2*self.GRID_SIZE - self.offset[1] + 1, self.GRID_SIZE - self.offset[0] + 1 : 2*self.GRID_SIZE - self.offset[0] + 1, : self.NUM_ROBOTS] = self.robot_positions
        
        
        self.obstacle_count = 0
    
        self.obstacles_on_robot_start = 0
        
        self.obstacles_on_robot_end = 0
        
        self.obstacles_off_robot_start = 0
        
        self.obstacles_off_robot_end = 0  

        return self.state


    def step (self, action):
        
        new_state = np.zeros(((2 * self.GRID_SIZE) + 2, (2 * self.GRID_SIZE) + 2, self.NUM_ROBOTS + 1), dtype=int)
        #self.obstacle_grid = self.state[self.GRID_SIZE - self.offset[1] + 1 : 2*self.GRID_SIZE - self.offset[1] + 1, self.GRID_SIZE - self.offset[0] + 1 : 2*self.GRID_SIZE - self.offset[0] + 1, self.NUM_ROBOTS]
        print(self.obstacle_grid)
        print("action: ", action)
        
        self.offset = self.obstacle_positions[self.obstacle_count]
        print("Initial Offset: ", self.offset)
        
        print("Obstacle State:")
        print(self.state[:,:,self.NUM_ROBOTS])
                
        
        if self.done:
            print("EPISODE COMPLETE")
        
        elif self.obstacle_count == self.GRID_SIZE  and self.iterations == self.GRID_SIZE - 2:
            self.done = True;
        
        #elif self.obstacle_count == 1:
            #self.done = True;
            
        elif self.obstacle_count == self.GRID_SIZE :
            self.iterations += 1
            self.obstacle_count = 0

        else:
            assert self.action_space.contains(action)      
        
            #TODO move obstacle
            
            #0 1 2
            #3 4 5
            #6 7 8
            
            x0 = self.offset[0]
            y0 = self.offset[1]
            
            #print("action: ", action)
            #print("x0: ", x0, " y0: ", y0)
            
            if action == 0:
                x = max(0,self.offset[0] - 1)
                y = max(0,self.offset[1] - 1)
            
            if action == 1:
                x = max(0,self.offset[0] - 1)
                y = self.offset[1]
            
            if action == 2:
                x = max(0,self.offset[0] - 1)
                y = min(self.GRID_SIZE - 1,self.offset[1] + 1)
                
            if action == 3:
                x = self.offset[0]
                y = max(0,self.offset[1] - 1)
            
            if action == 4:
                x = self.offset[0]
                y = self.offset[1]
                
            if action == 5:
                x = self.offset[0]
                y = min(self.GRID_SIZE - 1,self.offset[1] + 1)
            
            if action == 6:
                x = min(self.GRID_SIZE - 1,self.offset[0] + 1)
                y = max(0,self.offset[1] - 1)
                
            if action == 7:
                x = min(self.GRID_SIZE - 1,self.offset[0] + 1)
                y = self.offset[1]
            
            if action == 8:
                x = min(self.GRID_SIZE - 1,self.offset[0] + 1)
                y = min(self.GRID_SIZE - 1,self.offset[1] + 1)     
                
                
            self.obstacle_grid[self.offset[1]][self.offset[0]] -= 1
            self.obstacle_grid[y][x] += 1
            
            #print(self.obstacle_grid)
            
            self.obstacle_positions[self.obstacle_count] = [x,y]
            
            stacked = self.obstacle_grid[y][x] - 1
            
            new_obstacle_on_robot_start = 0
            new_obstacle_on_robot_end = 0
            new_obstacle_off_robot_start = 0
            new_obstacle_off_robot_end = 0
            for i in range(self.NUM_ROBOTS):
                if self.robot_positions[y][x][i] == 1:
                    new_obstacle_on_robot_start += 1
                elif self.robot_positions[y][x][i] == 2:
                    new_obstacle_on_robot_end += 1
                if self.robot_positions[y0][x0][i] == 1:
                    new_obstacle_off_robot_start += 1
                elif self.robot_positions[y0][x0][i] == 2:
                    new_obstacle_off_robot_end += 1
                    
            self.obstacles_on_robot_start += new_obstacle_on_robot_start
    
            self.obstacles_on_robot_end += new_obstacle_on_robot_end
            
            self.obstacles_on_robot_start -= new_obstacle_off_robot_start
    
            self.obstacles_on_robot_end -= new_obstacle_off_robot_end
            
            
            if self.obstacles_on_robot_start + self.obstacles_on_robot_end == 0:    
                if action != 4 or (self.iterations == 0 and self.iterations == 0):
                    s = Simulator.Simulation()
                    
                    s.set_render(False)
                    
                    for i in range(self.NUM_ROBOTS):  
                        s.add_robot_on_matrix(self.start_positions[i][0],self.start_positions[i][1], self.end_positions[i][0],self.end_positions[i][1], self.ROBOT_VEL, 10)
                        

                    s.add_obstacle_matrix(self.obstacle_grid)
                    
                    i = 0
                    
                    while i < self.MAX_STEPS and not all(a for a in s.arrived):
                        s.step_simulation()       
                        i += 1
                    
                    self.perc_speed = s.perc_speed
                    distance_travelled = s.distance_travelled
                    speed = s.speed         
                    arrived = s.arrived
                    
                    distance_to_goal = []
                    
                    for i in range(self.NUM_ROBOTS):
                        distance_to_goal.append(dist(self.start_positions[i],self.end_positions[i]))
                    
                    self.reward = 0.
                    
                    self.perc_dist = []
                    
                    for i in range(self.NUM_ROBOTS):
                        #print("i: ", i, " %: ", perc_speed[i], " arrived: ", 0.4*(1 - arrived[i]))
                        self.reward += 5*self.perc_speed[i]
                        if arrived[i]:
                            self.reward += 5 * distance_to_goal[i]/distance_travelled[i]
                            self.perc_dist.append(distance_travelled[i]/(distance_to_goal[i] * 50))
                        else:
                            self.reward -= 3
                    
                    #self.reward -= 10*stacked
                    
                    #if action == 4:
                        #self.reward -= 1
                    
                    if math.isnan(self.reward):
                        self.reward = 0.
                    
            else:
                self.reward = -100.
                self.done = True
                self.perc_dist = [-1]
            
            print("Reward: ",self.reward)
            
            #Set info
            self.info = {"reward": self.reward, "percentage_dist": self.perc_dist, "percentage_speed": self.perc_speed}
            
            self.obstacle_count += 1
            
            self.offset = self.obstacle_positions[self.obstacle_count]
            
            new_state[self.GRID_SIZE - self.offset[1] + 1 : 2*self.GRID_SIZE - self.offset[1] + 1, self.GRID_SIZE - self.offset[0] + 1 : 2*self.GRID_SIZE - self.offset[0] + 1, self.NUM_ROBOTS] = self.obstacle_grid
            new_state[self.GRID_SIZE - self.offset[1] + 1 : 2*self.GRID_SIZE - self.offset[1] + 1, self.GRID_SIZE - self.offset[0] + 1 : 2*self.GRID_SIZE - self.offset[0] + 1, : self.NUM_ROBOTS] = self.robot_positions    
            
            self.state = new_state
            
            
            print("After Action:")
            print(self.obstacle_grid)
            
            self.offset = self.obstacle_positions[self.obstacle_count]
            print("Final Offset: ", self.offset)

        try:
            assert self.observation_space.contains(self.state)
        except AssertionError:
            print("INVALID STATE", self.state)

        return [self.state, self.reward, self.done, self.info]


    def render (self, mode="human", save=False):

        s = Simulator.Simulation()
            
        s.set_render(True)
        
        for i in range(self.NUM_ROBOTS):  
            s.add_robot_on_matrix(self.start_positions[i][0],self.start_positions[i][1], self.end_positions[i][0],self.end_positions[i][1], self.ROBOT_VEL, 5)
            

        s.add_obstacle_matrix(self.obstacle_grid)
        
        if save: print("--------------------------------SAVE-----------------------------")
        
        for i in range(self.MAX_STEPS):
            s.step_simulation(save)


    def seed (self, seed=None):

        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def close (self):
        
        pass
    
    
if __name__ == "__main__":
    ray.init()           
            
    trainer = ppo.PPOTrainer(env=ObstaclesEnv, config={
        "framework": "torch",
        "num_workers": 12,
        "num_gpus": 1,
        "use_critic": True,
        "use_gae": True,
        "lambda": 0.95,
        "kl_coeff": 0.5,
        "rollout_fragment_length": 200,
        "train_batch_size": 4000,
        "sgd_minibatch_size": 128,
        "shuffle_sequences": True,
        "num_sgd_iter": 30,
        "lr": 5e-5, #1e-4
        "lr_schedule": None,
        "vf_loss_coeff": 1.0,
        "model": {
            "dim": 12,
            "conv_filters": [
            [25, [2, 2], 2],
            [25, [2, 2], 2],
            [25, [2, 2], 2],
            [25, [2, 2], 2],],
            "vf_share_layers": False,
        },
        "entropy_coeff": 0.01,
        "entropy_coeff_schedule": None,
        "clip_param": 0.1,
        "vf_clip_param": 10.0,
        "grad_clip": None,
        "kl_target": 0.01,
        "batch_mode": "truncate_episodes",
        "observation_filter": "NoFilter",
        "simple_optimizer": False,
        "_fake_gpus": False,})

########## FOR USE IF TRAINING STARTS FROM LATER CHACKPOINT ##################
    #cfg = {'env_config': {}, 'model': {}}
    #with open("/home/jenny/ray_results/PPO_ObstaclesEnv_2021-03-19_05-33-13a8ea2o3_/params.json") as json_file:
        #cfg = json.load(json_file)
        
    #print(cfg)
    #trial = 0
    #trainer.restore(str("/home/jenny/ray_results/PPO_ObstaclesEnv_2021-03-19_05-33-13a8ea2o3_/checkpoint_20/checkpoint-20"))
    #env = ObstaclesEnv(cfg)
    #env.seed(trial)
    #obs = env.reset()


    for i in range(500):
    # Perform one iteration of training the policy with PPO
        result = trainer.train()
        print(result)
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
