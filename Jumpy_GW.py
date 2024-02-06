
import numpy as np

class JumpyGW:
    
    # Initializing
    def __init__(self, size, number_obstacles):
        self.grid_size = size
        self.grid = np.zeros( (size,size) )
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.obstacles = set()
        self.agent_path = []

        for i in range(number_obstacles):
            obs_pos = ( np.random.randint(size), np.random.randint(size))
            self.obstacles.apend(obs_pos)


    # Reset the environemtn from scrach
    def reset(self):
        self.grid = np.zeros( (self.grid_size, self.grid_size) )
        self.agent_path = []
        return self.start

    # Verify the validity of a position in the grid        

