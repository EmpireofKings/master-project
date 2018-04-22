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
            if other.UP:
                return Point(self.X, self.Y - 1)
            elif other.RIGHT:
                return Point(self.X + 1, self.Y)
            elif other.DOWN:
                return Point(self.X, self.Y + 1)
            elif other.LEFT:
                return Point(self.X - 1, self.Y)
            else:
                raise NotImplementedError("Directions other than up, right, down and left are not implemented.")
        else:
            raise TypeError("Expecting Direction, got " + type(other))
    
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
        return "Drone %s at position "%self.id + self.position
    
    def move(self, direction):
        self._check_movability(direction)
        self.trace.append(self.position)
        self.position(self.position + direction)
        self.grid.position_drone(self)
            
    
    def get_position(self):
        return self.position
    
    def get_id(self):
        return self.id
    
    def _check_movability(self, direction):
        if self.grid.is_obsticle(self.position + direction):
            raise PositioningError(self + " hit an obsticle when moving " direction)
        # TODO: Other drones?


        
class Grid(object):

    def __init__(self, size_x, size_y):
        self.grid = np.array(size_x, size_y)
        
    def set_obsticles(self, seed, num_obsticles):
        '''Obstacles appear in grid as "O".'''
        # TODO
    
    def position_drone(self, drone):
        if self.is_obsticle(drone.get_position()):
            raise PositioningError(drone + " hit an obsticle.")
        
        p = drone.get_position()
        
        if self.grid[p.getX(), p.getY()] == None:
            self.grid[p.getX(), p.getY()] = drone.get_id()
        else:
            raise PositioningError("Location " + p + " is already used! Content: " + self.grid[p.getX(), p.getY()])
        
    
    def is_obsticle(self, point):
        if self.grid[point.getX(), point.getY] == "O"
            return True
        else:
            return False
        