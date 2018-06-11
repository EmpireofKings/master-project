# Simulator module
import numpy as np 

from enum import Enum
class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    
class PositioningError(Exception):
    pass

class TargetUnreachableError(Exception):
    pass

class Point(object):
    """2D Point"""
    def __init__(self, x, y):
        self.X = x
        self.Y = y
        
    def __str__(self):
        return "[X=%s, Y=%s]"%(self.X, self.Y)
    
    def __repr__(self):
        return "<X=%s, Y=%s>"%(self.X, self.Y)
    
    def __eq__(self, other):
        if isinstance(other, Point):
            return self.X == other.X and self.Y == other.Y
        else:
            return NotImplemented
    
    def __add__(self, other):
        """overload + operator"""
        if other is None:
            return self
        elif isinstance(other, Direction):
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
    
    def __hash__(self):
        return hash((self.X, self.Y))
    
    def get_x(self):
        return self.X
    
    def get_y(self):
        return self.Y
    
class Drone(object):
    
    def __init__(self, grid, position, id):
        assert isinstance(id, int), "Expecting int value as ID"
        self.position = position
        self.grid = grid
        self.id = id
        self.trace = []
        self.grid.position_drone(self)
        
    
    def __str__(self):
        return "Drone %s at position %s" % (self.id, self.position)
    
    def move(self, direction):
        self.grid.move_drone(self, direction)   # possibly throws exception
        self.trace.append(self.position)
        self.position = self.position + direction
            
    def observe_surrounding(self):
        # Order: Top, right, down, left
        surrounding = []
        for d in Direction:
            p = self.position + d
            surrounding.append(self.grid.get_value(p))
        return surrounding
    
    def get_position(self):
        return self.position
    
    def get_position_one_hot(self):
        pos_flat = self.position.get_y() * self.grid.size[1] + self.position.get_x()
        pos_one_hot = np.zeros(self.grid.size).ravel()
        pos_one_hot[pos_flat] = 1
        return pos_one_hot
        
    def get_id(self):
        return self.id
        
    def get_trace(self):
        return self.trace

        
class Grid(object):

    def __init__(self, size_y, size_x, seed):
        self._grid = np.full([size_y, size_x], None)
        self.size = (size_y, size_x)
        self.rs = np.random.RandomState(seed)
        self.drone = None
        self.are_drones_set = False
        self.are_obstacles_set = False
        
    def __str__(self):
        return "Simulator grid with size %s:\n%s" % (self.size, self._grid)
        
    def set_obstacles(self, num_obstacles):
        """Obstacles appear in grid as "O"."""
        i = 0
        while i < num_obstacles:
            y = self.rs.randint(low=0, high=self.size[0])
            x = self.rs.randint(low=0, high=self.size[1])
            try:
                self.set_value(Point(x, y), "O")
                i += 1
            except PositioningError:
                pass  # Try again
        self.are_obstacles_set = True
    
    def set_target(self):
        assert self.are_drones_set and self.are_obstacles_set, \
            "Obstacles and drones are needed to check if target is reachable."
        successful = False
        cells = self.asdict()
        while not successful:
            p = self.rs.choice(list(cells))
            if cells[p] is None and self._target_reachable(p):
                self.set_value(p, "T")
                successful = True
            else:
                del cells[p]
                if len(cells) == 0:
                    raise TargetUnreachableError("Too many obsticles to set reachable target")
    
    def asdict(self):
        """Returns a dict with cells as keys and content as value"""
        result = {}
        for y in range(self.size[0]):
            for x in range(self.size[1]):
                p = Point(x, y)
                result[p] = self.get_value(p)
        return result
    
    def _target_reachable(self, target):
        filled_grid = np.zeros(self.size) # Memory for visited cells
        return self._flood_fill(filled_grid, target)
        
        
    def _flood_fill(self, filled_grid, current_pos):
        """This is called recursive"""
        if isinstance(self.get_value(current_pos), int):
            # Found a drone
            return True
        elif self.get_value(current_pos) == "O":
            # Blocked by obstacle
            return False
        elif filled_grid[current_pos.get_y(), current_pos.get_x()] == 1:
            # Already visited this cell
            return False
        # Mark cell as visited
        filled_grid[current_pos.get_y(), current_pos.get_x()] = 1
        is_path = False
        for direction in Direction:
            if not is_path:
                is_path = self._flood_fill(filled_grid, current_pos + direction)
        return is_path
        
    def position_drone(self, drone):
        self.drone = drone
        p = drone.get_position()
        self.set_value(p, drone.get_id())
        self.are_drones_set = True
        
    def move_drone(self, drone, direction):
        p = drone.get_position()
        self.set_value(p + direction, drone.get_id())   # possibly throws exception
        self.reset_value(p)
        
    def get_value(self, point):
        x = point.get_x()
        y = point.get_y()
        if x < 0 or y < 0 or x >= self.size[1] or y >= self.size[0]:
            return "O"   # Tread cells outsied of grid borders as obsticles
        else:
            return self._grid[y, x] 
    
    def set_value(self, point, value):
        x = point.get_x()
        y = point.get_y()
        if x < 0 or y < 0 or x >= self.size[1] or y >= self.size[0]:
            raise IndexError("Index out of bound. [x=%s, y=%s" % (x, y))
        elif self._grid[y, x] is None:
            self._grid[y, x] = value
        else:
            raise PositioningError("Location %s is already used! Content: %s" % (point, self.get_value(point)))
        
    def reset_value(self, point):
        self._grid[point.get_y(), point.get_x()] = None
        
    def get_obsticles_vector(self):
        return (self._grid == "O").astype(int).ravel()
    
    def is_accessible(self, point):
        # Check for outside grid
        if point.get_x() < 0 or \
           point.get_y() < 0 or \
           point.get_y() > self.size[0] or \
           point.get_x() > self.size[1]:
            return False
        # Check for obstacles
        elif self.get_value(point) == "O":
            return False
        else:
            return True
        