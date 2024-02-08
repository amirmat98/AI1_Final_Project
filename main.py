
import numpy as np
import matplotlib.pyplot as plt
from Jumpy_GW import JumpyGW
from QLearning_Agent import QLearning_Agent
from Plotter import draw_agent_route, render_qvalue_heatmap, display_agent_policy_on_grid, refresh_visualization
from Globals_Variables import rewards, steps_episode
from Plotter import map_action_to_symbol


# Function to set up the simulation with a predefined environment and agent
def configure_simulation():
    # Seed the random number generator for reproducibility
    # temp_seed = np.random.randint(0, 100)
    # print("seed = ", temp_seed)
    # np.random.seed(temp_seed)
    np.random.seed(56)

    # Define parameters for the grid world and Q-learning agent
    params = {
        'grid_size': 10,         # Size of the grid world
        'number_obstacles': 5,        # Number of obstacles in the grid world
        'state_size': (10, 10), # Shape of the state space
        'number_actions': 5,          # Number of possible actions
        'learning_rate': 0.1,    # Learning rate for Q-learning updates
        'discount_factor': 0.9,  # Discount factor for future rewards
        'exploration_rate': 0.1, # Rate at which the agent explores the environment
        'episodes': 1000         # Number of episodes to run the simulation
    }

    # Initialize the grid world and agent with the specified parameters
    environment = JumpyGW(params['grid_size'], params['number_obstacles'])
    agent = QLearning_Agent(params['state_size'], params['number_actions'], params['learning_rate'], params['discount_factor'], params['exploration_rate'])
    agent.number_episodes = params['episodes']
    
    return environment, agent, params

# Function to run the simulation of the environment and agent interaction
def run_simulation(environment, agent, params):
    # Set up plotting
    figure, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes_list = axes.flatten()
    # ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    # Clear any previous metrics
    rewards.clear()
    steps_episode.clear()
    grid_layouts = []
    performance = []
    episode_path = []

    # Run the simulation for the specified number of episodes
    print("Ongoing training...")
    for episode_num in range(params['episodes']):
        current_state = environment.reset() # Reset the environment
        total_reward = 0 # Initialize total reward for the episode
        episode_qvalues = []
        
        agent.episode_path_reset() # Initialize agent's path for the new episode

        # Iterate until the agent reaches a final state
        while not environment.goal_check(current_state):
            action = agent.action_choose(current_state) # Agent selects an action
            next_state, reward = environment.action_execuation(current_state, action) # Environment responds
            agent.qtable_update(current_state, action, reward, next_state) # Agent updates its knowledge

            current_state = next_state # Move to the next state
            total_reward += reward # Accumulate rewards
            episode_qvalues.append(np.copy(agent.qtable[current_state]))
            agent.path_episode.append(current_state) # Record the new state

        # Record episode metrics
        rewards.append(total_reward)
        steps_taken = len(episode_qvalues)
        steps_episode.append(steps_taken)
        performance.append(total_reward / steps_taken)

        # Plot statistics after each episode
        grid_layout = np.zeros_like(environment.grid)
        obstacle_position = list(environment.obstacles)
        grid_layout[tuple(zip(*obstacle_position))] = -1
        grid_layout[environment.goal] = 0.5
        grid_layout[current_state] = 0.8
        grid_layouts.append(grid_layout)

        # Update episode_path
        episode_path = agent.path_episode              
        refresh_visualization(episode_num, agent, axes_list[0], axes_list[1], axes_list[2], axes_list[3], axes_list[4], axes_list[5], environment, episode_path)

        # Visualize the agent's trajectory path
        draw_agent_route(params['grid_size'], episode_path, axes_list[5])

    # Simulation is complete
    print("Training has been completed.")

    # Extract and display the optimal policy derived from Q-values
    optimal_policy = np.zeros_like(environment.grid, dtype = int)
    display_policy(optimal_policy)
    plt.show() # Display all the plots

# Function to display the optimal policy in a human-readable format
def display_policy(policy):
    print("\nOptimal Solution and Policy:")
    # Iterate through all states in the policy
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            action = policy[i, j]
            print(f"State ({i}, {j}): Move {map_action_to_symbol(action)}")

    print("\nOptimal Policy:")
    print(policy)

# Main entry point for the script
def main():
    # Configure the environment and agent, and fetch simulation parameters
    environment, q_agent, simulation_params = configure_simulation()
    # Run the simulation with the configured environment and agent
    run_simulation(environment, q_agent, simulation_params)

if __name__ == "__main__":
    main() # Execute the main function if the script is run directly
