class Obstacle:
    
    def __init__(self, x_size, y_size, pos, orientation, unit):
        e = 0.01 * unit
        self.poly = []
        self.poly.append( (0.5*x_size + e, 0.5*y_size + e) )
        self.poly.append( (0.5*x_size + e, - 0.5*y_size - e) )
        self.poly.append( (- 0.5*x_size - e, - 0.5*y_size - e) )
        self.poly.append( (- 0.5*x_size - e, 0.5*y_size + e) )
        self.x_size = x_size
        self.y_size = y_size
        self.p = pos
        self.r = orientation
