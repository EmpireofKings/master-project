# Unit tests for simulator

import unittest
import sys 
import os

# Import target class
sys.path.append(os.path.abspath("../../main/simulator"))
from Simulator import *


class PointTestCase(unittest.TestCase):
        
    def test_operator_overloading(self):
        p1 = Point(1,1)
        p2 = p1 + Direction.UP
        self.assertEqual(p2.getX(), 1)
        self.assertEqual(p2.getY(), 0)
        

class DroneTestCase(unittest.TestCase):

    def test_move_drone(self):
        #  Grid
        g1 = Grid(10, 5)
        g1.set_obsticles(1, 10)
        # Drone
        init_pos = Point(0,0)
        id = 1
        d1 = Drone(g1, init_pos, id)
        # Actions
        g1.position_drone(d1)
        d1.move(Direction.DOWN)
        # Assertions
        self.assertEqual(g1._grid[1,0], id)
        self.assertEqual((g1._grid == None).sum(), 10*5-10 - 1,
            "Expecting all entries but the drone and 10 obsticles to be None")
        self.assertEqual(d1.get_trace()[0], Point(0, 0), "Expecting point in trace")
        self.assertEqual(len(d1.get_trace()), 1, "Expecting just one entry in trace")
        
    def test_move_drone_to_obsticle(self):
        #  Grid
        g1 = Grid(10, 5)
        g1.set_obsticles(1, 10)
        # Drone
        init_pos = Point(0,0)
        id = 1
        d1 = Drone(g1, init_pos, id)
        # Actions
        g1.position_drone(d1)
        # Assertions
        with self.assertRaises(PositioningError) as context:
            d1.move(Direction.RIGHT)
        

class GridTestCase(unittest.TestCase):
    
    def test_initialization(self):
        g1 = Grid(size_y = 30, size_x = 20)
        self.assertEqual(g1._grid.shape, (30, 20))
        self.assertEqual((g1._grid == None).sum(), 30*20, "Expecting all entries to be None")
        
    def test_position_drone(self):
        g1 = Grid(300, 150)
        init_pos = Point(10,10)
        id = 0
        d1 = Drone(g1, init_pos, id)
        
        g1.position_drone(d1)
        self.assertEqual(g1._grid[10,10], id)
        self.assertEqual((g1._grid == None).sum(), 300*150-1, "Expecting all entries but one to be None")
        
    def test_set_obsticles(self):
        g1 = Grid(30, 15)
        g1.set_obsticles(1, 10)
        self.assertEqual((g1._grid == "O").sum(), 10, "Expecting 10 entries to be marked as obsticles")
        self.assertEqual((g1._grid == None).sum(), 30*15-10, "Expecting all entries but one to be None")

    def test_position_drone_on_obsticle(self):
        g1 = Grid(10, 5)
        g1.set_obsticles(1, 10)
        
        init_pos = Point(1,0) # Occupied by an obsticle
        id = 0
        d1 = Drone(g1, init_pos, id)
        
        self.assertEqual(g1._grid[0, 1], "O", "Tested point is expected to be an obsticle")
        self.assertFalse(g1.is_accessible(Point(1, 0)), "Test method 'is_obsticle'")
        with self.assertRaises(PositioningError) as context:
            g1.position_drone(d1)

if __name__ == '__main__':
    unittest.main()