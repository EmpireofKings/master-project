import numpy as np


from enum import Enum
class Direction(Enum):
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4
    
class PositioningError(Exception):
    pass

class Point(object):
    '''2D Point'''
    def __init__(self, x, y):
        self.X = x
        self.Y = y
        
    def __str__(self):
        return "[X=%s, Y=%s]"%(self.X, self.Y)
    
    def __add__(self, other):
        '''overload + operator'''
        if isinstance(other, Direction):
            if other == Direction.UP:
                return Point(self.X, self.Y - 1)
            elif other == Direction.RIGHT:
                return Point(self.X + 1, self.Y)
            elif other == Direction.DOWN:
                return Point(self.X, self.Y + 1)
            elif other == Direction.LEFT:
                return Point(self.X - 1, self.Y)
            else:
                raise NotImplementedError("Directions other than up, right, down and left are not implemented.")
        else:
            raise TypeError("Expecting Direction, got %s" % type(other))
    
    def getX(self):
        return self.X
    
    def getY(self):
        return self.Y
    

        
class Drone(object):
    
    def __init__(self, grid, position, id):
        self.position = position
        self.grid = grid
        self.id = id
        self.trace = []
        
    
    def __str__(self):
        return "Drone %s at position %s" % (self.id, self.position)
    
    def move(self, direction):
        #self._check_movability(direction)
        self.grid.move_drone(self, direction)
        self.trace.append(self.position)
        self.position = self.position + direction
            
    
    def get_position(self):
        return self.position
    
    def get_id(self):
        return self.id
        
    def get_trace(self):
        return self.trace
    
    def _check_movability(self, direction):
        if not self.grid.is_accessible(self.position + direction):
            raise PositioningError("%s hit an obsticle when moving %s" % (self, direction))
        # TODO: Other drones?


        
class Grid(object):

    def __init__(self, size_y, size_x):
        self._grid = np.full([size_y, size_x], None)
        self.size = (size_y, size_x)
        
    def set_obsticles(self, seed, num_obsticles):
        '''Obstacles appear in grid as "O".'''
        rs = np.random.RandomState(seed)
        i = 0
        while i < num_obsticles:
            y = rs.randint(low = 0, high = self.size[0])
            x = rs.randint(low = 0, high = self.size[1])
            p = Point(x, y)
            try:
                self.set_value(Point(x, y), "O")
                i += 1
            except PositioningError:
                pass  # Try again
        
    def position_drone(self, drone):
        p = drone.get_position()
        self.set_value(p, drone.get_id())
        
    def move_drone(self, drone, direction):
        p = drone.get_position()
        self.set_value(p + direction, drone.get_id())
        self.reset_value(p)
        
    def get_value(self, point):
        x = point.getX()
        y = point.getY()
        return self._grid[y, x]
    
    def set_value(self, point, value):
        x = point.getX()
        y = point.getY()
        if self._grid[y, x] == None:
            self._grid[y, x] = value
        else:
            raise PositioningError("Location %s is already used! Content: %s" % (point, self.get_value(point)))
        
    def reset_value(self, point):
        self._grid[point.getY(), point.getX()] = None
    
    def is_accessible(self, point):
        # Check for outside grid
        if point.getX() < 0 or \
           point.getY() < 0 or \
           point.getY() > self.size[0] or \
           point.getX() > self.size[1]:
            return False
        # Check for obstacles
        elif self.get_value(point) == "O":
            return False
        else:
            return True
        