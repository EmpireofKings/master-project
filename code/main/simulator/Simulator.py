# Simulator module
import numpy as np

from inspect import signature
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

    @staticmethod
    def fromtuple(position_tuple):
        return Point(position_tuple[0], position_tuple[1])
        
    def __str__(self):
        return "[X=%s, Y=%s]" % (self.X, self.Y)
    
    def __repr__(self):
        return "<X=%s, Y=%s>" % (self.X, self.Y)
    
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

    def __hash__(self):
        return hash(self.id)
    
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

    def observe_obstacles(self):
        return (np.array(self.observe_surrounding()) == "O").astype(np.int8)
    
    def get_position(self):
        return self.position

    def get_position_flat(self):
        return self.position.get_y() * self.grid.size[1] + self.position.get_x()
    
    def get_position_one_hot(self):
        pos_one_hot = np.zeros(self.grid.size, dtype=np.int8).ravel()
        pos_one_hot[self.get_position_flat()] = 1
        return pos_one_hot
        
    def get_id(self):
        return self.id
        
    def get_trace(self):
        return self.trace

        
class Grid(object):

    def __init__(self, size_y, size_x, target_seed, obstacles_seed):
        self._grid = np.full([size_y, size_x], None)
        self.discovery_map = np.zeros(size_y * size_x, dtype=np.int64)
        self.size = (size_y, size_x)
        self.rs_target = np.random.RandomState(target_seed)
        self.rs_obstacles = np.random.RandomState(obstacles_seed)
        self.drone = None
        self.are_drones_set = False
        self.are_obstacles_set = False
        
    def __str__(self):
        return "Simulator grid with size %s:\n%s" % (self.size, self._grid)
        
    def set_obstacles(self, num_obstacles):
        """Obstacles appear in grid as "O"."""
        i = 0
        while i < num_obstacles:
            # No obstacles at the borders
            y = self.rs_obstacles.randint(low=1, high=self.size[0]-1)
            x = self.rs_obstacles.randint(low=1, high=self.size[1]-1)
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
            p = self.rs_target.choice(list(cells))
            if cells[p] is None and self._target_reachable(p):
                self.set_value(p, "T")
                successful = True
            else:
                del cells[p]
                if len(cells) == 0:
                    raise TargetUnreachableError("Too many obstacles to set reachable target")
    
    def asdict(self):
        """Returns a dict with cells as keys and content as value"""
        result = {}
        for y in range(self.size[0]):
            for x in range(self.size[1]):
                p = Point(x, y)
                result[p] = self.get_value(p)
        return result
    
    def _target_reachable(self, target):
        filled_grid = np.zeros(self.size)  # Memory for visited cells
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
        self.discovery_map += drone.get_position_one_hot()
        self.are_drones_set = True
        
    def move_drone(self, drone, direction):
        p = drone.get_position()
        p_new = p + direction
        try:
            self.set_value(p_new, drone.get_id())   # possibly throws exception
            self.reset_value(p)
            p_new_flat = p_new.get_y() * self.size[1] + p_new.get_x()
            self.discovery_map[p_new_flat] += 1  # drone.get_position_one_hot() not possible: position not yet updated
        except IndexError as e:
            p_flat = p.get_y() * self.size[1] + p.get_x()
            self.discovery_map[p_flat] += 1  # drone.get_position_one_hot() not possible: position not yet updated
            raise e
        except PositioningError as e:
            p_flat = p.get_y() * self.size[1] + p.get_x()
            self.discovery_map[p_flat] += 1  # drone.get_position_one_hot() not possible: position not yet updated
            raise e
        
    def get_value(self, point):
        x = point.get_x()
        y = point.get_y()
        if x < 0 or y < 0 or x >= self.size[1] or y >= self.size[0]:
            return "O"   # Treat cells outside of grid borders as obstacles
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
        
    def get_obstacles_flat(self):
        return (self._grid == "O").astype(int).ravel()

    def get_discovery_map_flat(self):
        return self.discovery_map.copy().ravel()
    
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


class Simulator(object):

    def __init__(self, drone_id, grid_size):
        self.drone_id = drone_id
        self.grid_size = grid_size
        self.step_num = 0
        self.grid = None
        self.drones = None
        self.reward_fkt = None
        self.get_state = None

    def build_world(self, initial_position, num_obstacles, target_seed, obstacles_seed):

        self.grid = Grid(self.grid_size[0], self.grid_size[1], target_seed, obstacles_seed)
        self.grid.set_obstacles(num_obstacles)

        self.drone = Drone(self.grid, position=initial_position, id=self.drone_id)

        self.grid.set_target()

        self.step_num = 0

    def define_state(self,
                     drone_location=True,
                     discovery_map=False,
                     drone_observed_obstacles=False):

        if drone_location and not discovery_map and not drone_observed_obstacles:
            def get_state():
                return self.drone.get_position_one_hot()

        elif drone_location and discovery_map and not drone_observed_obstacles:
            def get_state():
                return np.concatenate((self.drone.get_position_one_hot(),
                                       self.grid.get_discovery_map_flat()))

        elif drone_location and discovery_map and drone_observed_obstacles:
            def get_state():
                return np.concatenate((self.drone.get_position_one_hot(),
                                       self.grid.get_discovery_map_flat(),
                                       self.drone.observe_obstacles()))
        else:
            raise ValueError("State definition not covered by simulator")

        self.get_state = get_state

    def set_reward_function(self, reward_fkt):
        allowed_params = ["drone", "move_direction", "discovery_map", "step_num"]
        param_names = signature(reward_fkt).parameters
        for param_name in param_names:
            assert param_name in allowed_params, "Reward function has invalid parameter name. Allow are %s"\
                                                 % allowed_params

        self.reward_fkt = reward_fkt

    def step(self, action_idx):
        assert isinstance(action_idx, int), "got %s with value %s" % (type(action_idx), action_idx)

        if self.reward_fkt is not None:

            result = self.reward_fkt(drone=self.drone,
                                     move_direction=Direction(action_idx),
                                     discovery_map=self.grid.get_discovery_map_flat(),
                                     step_num=self.step_num)

            assert isinstance(result, tuple), "Expecting reward function to return tuple of (reward, done)"

            reward, done = result

            assert reward is not None, "Return function returned None reward for drone %s and action index %s"\
                                       % (self.drones[id], a)
            assert done is not None, "Expect reward function to return True if episode ended or False otherwise." \
                                     "Returned None for drone %s and action index %s" % (self.drones[id], a)

            next_state = self.get_state()

            self.step_num += 1

            return next_state, reward, done
        else:
            raise ValueError("Reward function not specified.")
