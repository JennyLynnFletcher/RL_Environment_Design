import math

e = 1

def dist(pos0, pos1):
    return math.sqrt((pos0[0]-pos1[0])**2 + (pos0[1]-pos1[1])**2)

class Agent:    
    def __init__(self, agent_id, position, goal_location, preferred_speed, radius, max_velocity = None, max_acceleration = None):
        self.agent_id = agent_id
        self.p = position
        self.g = goal_location
        self.s_pref = preferred_speed
        self.v_max = max_velocity
        self.a_max = max_acceleration
        self.v = [0., 0.]
        self.CRVO = None
        self.apexes = []
        self.poly = []
        self.radius = radius
        self.construct_poly(self.radius + e)
        self.v_pref = [0.,0.]
        self.update_preferred_velocity()
        self.arrived = False
        
    def clamp(self, n, smallest, largest): 
        return max(smallest, min(n, largest))
     
    def construct_poly(self, r):
        #approximates the robot as a regular octagon of radius r
        self.poly = []
        self.poly.append([0, r])
        self.poly.append([(r/math.sqrt(2)), (r/math.sqrt(2))])
        self.poly.append([r, 0])
        self.poly.append([(r/math.sqrt(2)), - (r/math.sqrt(2))])
        self.poly.append([0, - r])
        self.poly.append([- (r/math.sqrt(2)), - (r/math.sqrt(2))])
        self.poly.append([- r, 0])
        self.poly.append([- (r/math.sqrt(2)), (r/math.sqrt(2))])        
        
    def move(self, vec, timestep):
        new_pos = [self.p[0] + (vec[0] * timestep), self.p[1] + (vec[1] * timestep)]
        self.p = new_pos
        self.update_preferred_velocity()
        
        diff = dist(self.g, new_pos)
        
        if abs(diff) < 0.5*self.radius:
            self.arrived = True
            self.v_pref = [0.,0.] 
            self.v = [0., 0.]
            
        return self.arrived
        
    def set_CRVO(self, CRVO):
        self.CRVO = CRVO
        
    def set_apexes(self, apexes):
        self.apexes = apexes
    
    def update_preferred_velocity(self):
        v_pref_direction_unnormalised = ( (self.g[0] - self.p[0], self.g[1] - self.p[1]) )
        magnitude = math.sqrt(v_pref_direction_unnormalised[0]**2 + v_pref_direction_unnormalised[1]**2)
        if magnitude != 0 :
            v_pref_direction_normalised = (v_pref_direction_unnormalised[0]/magnitude, v_pref_direction_unnormalised[1]/magnitude)
        else:
            v_pref_direction_normalised = (0,0)
        if magnitude > self.s_pref:
            self.v_pref = (v_pref_direction_normalised[0]*self.s_pref, v_pref_direction_normalised[1]*self.s_pref)
        else:
            #self.v_pref = v_pref_direction_unnormalised
            self.v_pref = (v_pref_direction_normalised[0]*self.s_pref, v_pref_direction_normalised[1]*self.s_pref)
        
    def update_velocity(self, velocity):
        self.v = velocity
        
        
