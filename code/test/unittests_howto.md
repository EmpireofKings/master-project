# How to run python unit tests

Open a shell and change to the directiory of the test file.

Execute:

    python -m unittest -v Simulator_tests.py
    
Example output:
    
    test_move_drone (Simulator_tests.DroneTestCase) ... ok
    test_initialization (Simulator_tests.GridTestCase) ... ok
    test_position_drone (Simulator_tests.GridTestCase) ... ok
    test_position_drone_on_obsticle (Simulator_tests.GridTestCase) ... ok
    test_set_obsticles (Simulator_tests.GridTestCase) ... ok
    test_operator_overloading (Simulator_tests.PointTestCase) ... ok

    ----------------------------------------------------------------------
    Ran 6 tests in 0.000s

    OK