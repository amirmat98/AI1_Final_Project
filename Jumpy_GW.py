
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
    def position_validation(self, position):
        if ( (0 <= position[0] < self.grid_size) and (0 <= position[1] < self.grid_size) ):
            return True
        else:
            return False
    
    # Determine whether a given position constitutes an obstacle
    def obstacle_check(self, position):
        if (position in self.obstacles):
            return True
        else:
            return False
    
    # Determine whether a given position represents the goal state
    def goal_check(self, position):
        if (position == self.goal):
            return True
        else:
            return False

    # Action Effect Dictionary Definiation
    Action_Effect = {
        0: (-1, 0), # UP
        1: (1, 0), # Down
        2: (0, -1), # Left
        3: (0, 1), #Right
    }
    
    # Execute the designated action and provide the updated position and reward
    def action_execuation(self, position, action):
        if action in self.Action_Effect:
            delta = self.Action_Effect[action]
            new_positon = (position[0] + delta[0], position[1] + delta[1])
        else:
            jump_direction = np.random.choice([-1,1], size = 2)
            new_positon = (position[0] + jump_direction[0], position[1] + jump_direction[1])

        # update postion if valid and not an obstacle
        new_positon = self.update_position(position, new_positon)

        # update grid
        self.update_grid(position, new_positon)

        # calculate reward
        reward = self.calculate_reward(new_positon)

        # add the current position to the agent's path
        self.agent_path.append(new_positon)

        return new_positon, reward
    

    # define update position
    def update_position(self, position, new_positon):
        if self.position_validation(new_positon) and not self.obstacle_check(new_positon):
            return new_positon
        else:
            return position
    
    # define update grid
    def update_grid(self, position, new_positon):
        self.grid[position] = 0
        self.grid[new_positon] = 1
    
    # define calculation reward
    def calculate_reward(self, position):
        if self.goal_check(position):
            return 10
        else:
            return -1
    