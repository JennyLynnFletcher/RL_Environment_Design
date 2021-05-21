import math

class Cone:
    
    def __init__(self, apex,min_point, max_point):
         self.apex = apex
         self.min_point = min_point
         self.max_point = max_point
         
    def dist(self, pos0, pos1):
        return math.sqrt((pos0[0]-pos1[0])**2 + (pos0[1]-pos1[1])**2)
            
    def contains(self, point):
        left_min  = ((self.apex[0] - self.min_point[0])*(point[1] - self.min_point[1]) - (self.apex[1] - self.min_point[1])*(point[0] - self.min_point[0]) < 0)
        right_max = ((self.apex[0] - self.max_point[0])*(point[1] - self.max_point[1]) - (self.apex[1] - self.max_point[1])*(point[0] - self.max_point[0]) > 0)        
        return (left_min and right_max)
