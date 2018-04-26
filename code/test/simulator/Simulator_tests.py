# Unit tests for simulator

import unittest
import sys 
import os
import random

# Import target module
sys.path.append(os.path.abspath("../../main/simulator"))
from Simulator import *

SEED = 1


class PointTestCase(unittest.TestCase):
        
    def test_operator_overloading(self):
        p1 = Point(1,1)
        p2 = p1 + Direction.UP
        self.assertEqual(p2.getX(), 1)
        self.assertEqual(p2.getY(), 0)
        
    def test_operator_overloading_none(self):
        p1 = Point(1,1)
        p2 = p1 + None
        self.assertEqual(p2, p1, "Expecting point not to change if direction is None")
        

class DroneTestCase(unittest.TestCase):

    def test_init_drone(self):
        #  Grid
        g1 = Grid(10, 5, SEED)
        g1.set_obsticles(10)
        # Drone
        init_pos = Point(0,0)
        id = 1
        d1 = Drone(g1, init_pos, id)
        # Assertions
        self.assertEqual(g1._grid[0,0], id)
        self.assertEqual((g1._grid == None).sum(), 10*5-10 - 1,
            "Expecting all entries but the drone and 10 obsticles to be None")
        self.assertEqual(len(d1.get_trace()), 0, "Expecting trace to be empty")
        
    def test_init_drone_on_obsticle(self):
        g1 = Grid(10, 5, SEED)
        g1.set_obsticles(10)
        
        init_pos = Point(1,0) # Occupied by an obsticle
        id = 1
        
        self.assertEqual(g1._grid[0, 1], "O", "Tested point is expected to be an obsticle")
        self.assertFalse(g1.is_accessible(Point(1, 0)), "Test method 'is_obsticle'")
        with self.assertRaises(PositioningError) as context:
            d1 = Drone(g1, init_pos, id)

    def test_move_drone(self):
        #  Grid
        g1 = Grid(10, 5, SEED)
        g1.set_obsticles(10)
        # Drone
        init_pos = Point(0,0)
        id = 1
        d1 = Drone(g1, init_pos, id)
        # Actions
        d1.move(Direction.DOWN)
        # Assertions
        self.assertEqual(g1._grid[1,0], id)
        self.assertEqual((g1._grid == None).sum(), 10*5-10 - 1,
            "Expecting all entries but the drone and 10 obsticles to be None")
        self.assertEqual(d1.get_trace()[0], Point(0, 0), "Expecting point in trace")
        self.assertEqual(len(d1.get_trace()), 1, "Expecting just one entry in trace")
        
    def test_move_drone_to_obsticle(self):
        #  Grid
        g1 = Grid(10, 5, SEED)
        g1.set_obsticles(10)
        # Drone
        init_pos = Point(0,0)
        id = 1
        d1 = Drone(g1, init_pos, id)
        # Assertions
        with self.assertRaises(PositioningError) as context:
            d1.move(Direction.RIGHT)
        
    def test_observe_surrounding(self):
        #  Grid
        g1 = Grid(10, 5, SEED)
        g1.set_obsticles(10)
        # Drone
        init_pos = Point(0,0)
        id = 1
        d1 = Drone(g1, init_pos, id)
        # Actions
        suroundings = d1.observe_surrounding()
        # Assertions
        self.assertEqual(len(suroundings), 4, "Expecting four surounding cells")
        self.assertEqual(suroundings[0], "O", "Expecting obsticle.")
        self.assertEqual(suroundings[1], "O", "Expecting obsticle.")
        self.assertEqual(suroundings[2], None, "Expecting empty cell")
        self.assertEqual(suroundings[3], "O", "Expecting obsticle.")
        

class GridTestCase(unittest.TestCase):
    
    def test_init_grid(self):
        g1 = Grid(size_y = 30, size_x = 20, seed=SEED)
        self.assertEqual(g1._grid.shape, (30, 20))
        self.assertEqual((g1._grid == None).sum(), 30*20, "Expecting all entries to be None")
        
    def test_set_obsticles(self):
        g1 = Grid(30, 15, SEED)
        g1.set_obsticles(10)
        self.assertEqual((g1._grid == "O").sum(), 10, "Expecting 10 entries to be marked as obsticles")
        self.assertEqual((g1._grid == None).sum(), 30*15-10, "Expecting all entries but one to be None")
           
    def test_set_reachable_target(self):
        # Grid
        g1 = Grid(10, 5, seed=2)
        g1.set_obsticles(10)
        # Drone
        init_pos = Point(0,0)
        id = 99
        d1 = Drone(g1, init_pos, id)
        # Target
        g1.set_target() # Target will be in (2, 1) because of seed
        # Assertions
        self.assertEqual(g1.get_value(Point(2, 1)), "T", "Tested point is expected to be an target")
        self.assertEqual((g1._grid == "T").sum(), 1, "Expecting one entrie to be marked as target")
        self.assertEqual((g1._grid == None).sum(), 10*5-10-1-1, "Expecting all other entries to be None")
        
    def test_set_reachable_target_too_many_obsticles(self):
        # Grid
        g1 = Grid(10, 5, seed=2)
        g1.set_obsticles(10*5 - 5)
        # Drone
        init_pos = Point(3,0)
        id = 99
        d1 = Drone(g1, init_pos, id)
        # Assertions
        with self.assertRaises(TargetUnreachableError) as context:
            g1.set_target()
        
    def test_set_reachable_target_many_obsticles(self):
        # Grid
        g1 = Grid(10, 5, seed=2)
        g1.set_obsticles(10*5 - 20)
        # Drone
        init_pos = Point(1,0)
        id = 99
        d1 = Drone(g1, init_pos, id)
        # Actions
        g1.set_target()
        # Assertions
        self.assertEqual(g1.get_value(Point(0, 0)), "T", "Tested point is expected to be an target")

        
class SimulatorTestCase(unittest.TestCase):
    
    def test_one_drone_random_walk(self):
        # Grid
        g1 = Grid(10, 5, seed=2)
        g1.set_obsticles(10)
        # Drone
        init_pos = Point(0,0)
        id = 99
        d1 = Drone(g1, init_pos, id)
        # Target
        g1.set_target() # Target will be in (2, 1) because of seed
        # Actions
        random.seed(6)
        while not "T" in d1.observe_surrounding():
            rand_direction = random.choice(list(Direction))
            try:
                d1.move(rand_direction)
                #print("\n", g1)
            except (PositioningError, IndexError):
                pass    # Try again
            
            
        # Assertions
        self.assertEqual(d1.get_trace()[0], Point(0, 0), "Expecting initial point in trace")
        self.assertEqual(len(d1.get_trace()), 12, "Expecting 12 entries in trace with seed=6")
    
    
    def test_three_drones_random_walk(self):
        # Grid
        g1 = Grid(10, 5, seed=2)
        g1.set_obsticles(10)
        # Drones
        d1 = Drone(g1, Point(0,0), 0)
        d2 = Drone(g1, Point(0,1), 1)
        d3 = Drone(g1, Point(0,2), 2)
        # Target
        g1.set_target() # Target will be in (2, 1) because of seed
        # Actions
        random.seed(6)
        while True:
            common_surrounding = []
            for drone in [d1, d2, d3]:
                rand_direction = random.choice(list(Direction))
                try:
                    drone.move(rand_direction)
                except (PositioningError, IndexError):
                    #  Reward function here...
                    pass    # Try again
                common_surrounding += drone.observe_surrounding()
            #print("\n", g1, "\ncommon_surrounding:\n", common_surrounding)
            if "T" in common_surrounding:
                break
            
        # Assertions
        self.assertEqual(d1.get_trace()[0], Point(0, 0), "Expecting initial point in trace")
        self.assertEqual(len(d2.get_trace()), 2, "Expecting 2 entries in trace with seed=6")
        
        
if __name__ == '__main__':
    unittest.main()