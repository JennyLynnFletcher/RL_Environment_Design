import torch
import math
import time
import Agent
import Obstacle
import RVO as RVO

import matplotlib.pyplot as plt
import random
import datetime
from colour import Color
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'serif': 'cm'})
rc('text', usetex=True)


e = 0.01

white = (255,255,255)
black = (0,0,0)
red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)
grey = (100,100,100)

gradient_0 = list(Color("green").range_to(Color("blue"),500))
gradient_1 = list(Color("orange").range_to(Color("red"),500))


def dist(pos0, pos1):
    return math.sqrt((pos0[0]-pos1[0])**2 + (pos0[1]-pos1[1])**2)

class Simulation:
    
    def __init__(self):
        self.timestep = 1./100.
        self.border = 100
        self.unit = 50
        
        self.agents = []
        self.arrived = []
        self.obstacles = []
        
        self.perc_speed = []
        self.distance_travelled = []
        self.speed = []
        self.steps = 0
        
        self.path = []
        
        self.render = False
        self.saved = False
        
    def set_render(self, set_render):
        if set_render:
            global pygame
            pygame = __import__('pygame', globals(), locals()) 
            #import pygame
            pygame.init()
            self.gameDisplay = pygame.display.set_mode((1000,1000))
            self.gameDisplay.fill(white)
            self.pixAr = pygame.PixelArray(self.gameDisplay)
            self.render = True
            #pygame.display.update()
        
    def step_simulation(self, save=False):
        self.steps += 1
        RVO.CRVO(self.agents, self.obstacles)
        for j, (agent, arrived) in enumerate(zip(self.agents, self.arrived)):
            if not arrived:
                agent.update_preferred_velocity()
                CRVO = agent.CRVO
                agent.set_CRVO(CRVO)
                v_pref  = agent.v_pref
                apexes = agent.apexes
                agent.set_apexes(apexes)
                vel, _, _ = RVO.find_velocity(agent, CRVO, apexes, v_pref, agent.s_pref)
                agent.update_velocity(vel)
                if agent.move(vel, self.timestep):
                    self.set_arrived(j)
                
                speed = dist([0,0], vel)
                self.record_perc_speed(j, speed)
                self.record_distance_travelled(j, speed)
                self.record_speed(j, speed)
            self.path[j].append(agent.p)
        
        if self.render:
            self.gameDisplay.fill(white)
            for agent in self.agents:
                if agent.agent_id == 0:
                    pygame.draw.circle(self.gameDisplay, black, (agent.p[0] + self.border, agent.p[1] + self.border), agent.radius)
                    #pygame.draw.circle(self.gameDisplay, red, (agent.g[0] + self.border, agent.g[1] + self.border), agent.radius)                    
                else:
                    pygame.draw.circle(self.gameDisplay, black, (agent.p[0] + self.border, agent.p[1] + self.border), agent.radius)
                    #pygame.draw.circle(self.gameDisplay, 'coral', (agent.g[0] + self.border, agent.g[1] + self.border), agent.radius)
                
            for obstacle in self.obstacles:
                pygame.draw.rect(self.gameDisplay, grey, (obstacle.p[0] - 0.5*obstacle.x_size + self.border, obstacle.p[1] - 0.5*obstacle.y_size + self.border, obstacle.x_size, obstacle.y_size), width=0)
            pygame.display.update()
            
            
            if save and (all(self.arrived) or self.steps == 500) and not self.saved:
                print(self.steps)
                self.gameDisplay.fill(white)
                for obstacle in self.obstacles:
                    pygame.draw.rect(self.gameDisplay, grey, (obstacle.p[0] - 0.5*obstacle.x_size + self.border, obstacle.p[1] - 0.5*obstacle.y_size + self.border, obstacle.x_size, obstacle.y_size), width=0)
                
                pygame.draw.circle(self.gameDisplay, gradient_0[0].hex_l, (self.path[0][0][0] + self.border, self.path[0][0][1] + self.border), agent.radius)
                pygame.draw.circle(self.gameDisplay, gradient_0[0].hex_l, (self.path[1][0][0] + self.border, self.path[1][0][1] + self.border), agent.radius)
                for s in range(self.steps):
                    pygame.draw.circle(self.gameDisplay, gradient_0[min(s,499)].hex_l, (self.path[0][s][0] + self.border, self.path[0][s][1] + self.border), 1)
                    pygame.draw.circle(self.gameDisplay, gradient_0[min(s,499)].hex_l, (self.path[1][s][0] + self.border, self.path[1][s][1] + self.border), 1)
                pygame.draw.circle(self.gameDisplay, gradient_0[499].hex_l, (self.agents[0].g[0] + self.border, self.agents[0].g[1] + self.border), agent.radius)
                pygame.draw.circle(self.gameDisplay, gradient_0[499].hex_l, (self.agents[1].g[0] + self.border, self.agents[1].g[1] + self.border), agent.radius)
                    
                filename = '/home/jenny/Documents/Part II Project/Code/Images/%s.bmp' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                pygame.image.save(self.gameDisplay, filename)
                print("Image has been saved as %s" % filename)
                self.saved = True
            
            #time.sleep(self.timestep)
        
    def add_robot(self, position, goal_location, preferred_speed, radius = 10, max_velocity = None, max_acceleration = None):
        self.agents.append(Agent.Agent(len(self.agents), position, goal_location, preferred_speed, radius, max_velocity = None, max_acceleration = None))
        self.arrived.append(False)
        self.perc_speed.append(0)
        self.distance_travelled.append(0)
        self.path.append([])
        self.speed.append([])
    
    def add_obstacle(self,x_size, y_size, pos, orientation):
        self.obstacles.append(Obstacle.Obstacle(x_size, y_size, pos, orientation, self.unit))
        
    def add_obstacle_matrix(self, matrix):
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[j][i] >= 1:
                    self.add_obstacle(self.unit, self.unit, [self.unit*i +  self.border, self.unit*j + self.border], [0,0,0,1])
                    
    def add_robot_on_matrix(self, position_x, position_y, goal_x, goal_y, preferred_speed, radius = 10, max_velocity = None, max_acceleration = None):
        self.add_robot([self.unit * position_x + self.border, self.unit * position_y + self.border], [self.unit * goal_x + self.border, self.unit * goal_y + self.border], preferred_speed, radius, max_velocity, max_acceleration)
                    
    def set_arrived(self, agent):
        self.arrived[agent] = True
                    
    def plot_RVO(self, agent):
        plt.clf()
        CRVO = self.agents[agent].CRVO
        apexes = self.agents[agent].apexes
        v_pref  = self.agents[agent].v_pref
        vel, cones, vs = RVO.find_velocity(self.agents[agent], CRVO, apexes, v_pref, self.agents[agent].s_pref)
            
        
        plt.plot(v_pref[0], v_pref[1], 'ro')
        plt.plot(vel[0], vel[1], 'go')
        
        #for v in vs:
            #plt.plot(v[0], v[1], 'go')
        
        for cone, points in zip(cones, CRVO):
            colour = (random.random(), random.random(), random.random())
            plt.plot(cone.apex[0], cone.apex[1], 'bo')
            plt.plot([cone.apex[0], cone.min_point[0]], [cone.apex[1], cone.min_point[1]], color = colour)
            plt.plot([cone.apex[0], cone.max_point[0]], [cone.apex[1], cone.max_point[1]], color = colour, linestyle = '--')
            if len(points) > 0:
                plt.scatter(*zip(*points), color = colour, marker = '.', s=1)
        
        plt.xlabel("x component of velocity")
        plt.ylabel("y component of velocity")
        plt.savefig("Images/poly.pdf")
        print("saved " + str(i))
        
    def record_perc_speed(self, agent, speed):
        #preferred_speed = self.agents[agent].s_pref        
        #self.perc_speed[agent].append(speed/preferred_speed)
        #print(self.perc_speed)
        preferred_speed = self.agents[agent].s_pref 
        self.perc_speed[agent] = (self.perc_speed[agent] * (self.steps - 1) + (speed/preferred_speed))/self.steps
    
    def record_distance_travelled(self, agent, speed):
        self.distance_travelled[agent] += speed * self.timestep
    
    def record_speed(self, agent, speed):      
        self.speed[agent].append(speed)
     
if __name__ == "__main__":     
    s = Simulation()

    s.add_robot_on_matrix(0,0, 4,1, 200, 6)
    s.add_robot_on_matrix(2,4, 1,0, 200, 6)
    

    s.add_obstacle_matrix( [[0, 0, 1, 0, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 1],
                            [1, 1, 0, 0, 1]])
    
    #s.add_obstacle_matrix( [[0, 0, 0, 0, 0],
                            #[0, 0, 0, 0, 0],
                            #[0, 0, 0, 0, 0],
                            #[0, 0, 0, 0, 0],
                            #[0, 0, 0, 0, 0]])

    s.set_render(True)
    for i in range(10000):   
        #print("-------------------------------------------------------------------------")    
        s.step_simulation(save=True)

        if i%100 == 0 and not s.arrived[0]:
            s.plot_RVO(0)
     
    
