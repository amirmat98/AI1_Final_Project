import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from Globals_Variables import rewards, steps_episode


# Function to convert actions into visual symbols
def map_action_to_symbol(action):
    arrow_map = {
        0: '↑',  # Up
        1: '↓',  # Down
        2: '←',  # Left
        3: '→',  # Right
        4: '⇆'   # Jump
    }
    # Fallback to '?' when the action is not found in the mapping
    if action in arrow_map:
        return arrow_map.get(action)
    else:
        return '?'


# Function to plot the agent's path on the grid
def draw_agent_route (size, agent_path, axis):
    axis.clear()
    axis.set_title(r'$\bf{Agent\'s\ Trajectory\ Path}$', fontsize=14)
    axis.set_xlabel('Grid Width')
    axis.set_ylabel('Grid Height')
    axis.set_xticks(np.arange(size+1))  # Add +1 to include the edge for better visualization
    axis.set_yticks(np.arange(size+1))  # Add +1 to include the edge for better visualization
    axis.grid(which='both', color='red', linestyle='-', linewidth=2)
    x_coords = [coord[0] for coord in agent_path]
    y_coords = [coord[1] for coord in agent_path]
    axis.plot(np.array(y_coords)+0.5, np.array(x_coords)+0.5, marker='o', linestyle='-', color='blue')  # Add +0.5 to center the markers in the cells
    axis.invert_yaxis()  # Invert y-axis to match the visualization of the grade


# Function to create a heatmap representation of the highest Q-values
def render_qvalue_heatmap(q_values, axis):
    axis.clear()
    optimal_qvalues = np.max(q_values, axis=2)
    heatmap = axis.matshow(optimal_qvalues, cmap='cool')
    colorbar = plt.colorbar(heatmap, ax=axis)
    colorbar.ax.set_ylabel('Q-values', rotation=-90, va="bottom")
    axis.set_title('Heatmap of Optimal Q-values', pad=20)
    axis.set_xlabel('Width of Grid')
    axis.set_ylabel('Height of Grid')
    axis.set_xticks(np.arange(q_values.shape[1]))
    axis.set_yticks(np.arange(q_values.shape[0]))


# Function to display the agent's policy on a grid
def display_agent_policy_on_grid(q_values, axis):
    axis.clear()
    # Determine the best action (highest Q value) for each state
    optimal_actions = np.argmax(q_values, axis=2)

    action_values_mapping = {
        0: 0.25,  # Assign 0.25 to 'Up' action
        1: 0.50,  # Assign 0.50 to 'Down' action
        2: 0.75,  # Assign 0.75 to 'Left' action
        3: 1.0,   # Assign 1.0 to 'Right' action
        4: 0.0    # Assign 0.0 to 'Jump' action
    }   

    # Map the optimal actions to their corresponding visualization values
    policy_display_values = np.vectorize(action_values_mapping.get, otypes=[float])(optimal_actions)

    # Create a matrix display of the policy values with a color map
    color_axis = axis.matshow(policy_display_values, cmap='cool', aspect='equal')

    # Configure the color map and normalization for the color bar
    color_map = plt.cm.cool
    color_norm = Normalize(vmin=0.0, vmax=1.0)

    # Create a ScalarMappable for the color bar
    scalar_mapper = ScalarMappable(cmap=color_map, norm=color_norm)
    scalar_mapper.set_array([])

    # Add a color bar with specific tick labels for each action
    color_bar = plt.colorbar(scalar_mapper, ax=axis, ticks=[0.0, 0.25, 0.50, 0.75, 1.0])
    color_bar.ax.set_yticklabels(['Jump', 'Up', 'Down', 'Left', 'Right'])  # Label the ticks with action names

    # Overlay action arrows on the grid
    for row_index, row in enumerate(optimal_actions):
        for col_index, action_code in enumerate(row):
            action_arrow = action_values_mapping(action_code)
            axis.text(col_index, row_index, action_arrow, horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')

    # Set grid lines centered on each cell edge
    axis.set_xticks(np.arange(optimal_actions.shape[1]) - 0.5, minor=True)
    axis.set_yticks(np.arange(optimal_actions.shape[0]) - 0.5, minor=True)
    
    # Draw the grid lines
    axis.grid(which='minor', color='black', style='-', linewidth=2)
    
    # Remove tick marks
    axis.tick_params(axis='both', which='both', length=0)
    
    # Set the title for the axes
    axis.set_title('Optimal Policy Visualization\n')



# Function to update visualization plots after training episodes
def refresh_visualization(episode_number, agent, axis_qvalues, axis_policy, axis_path, axis_steps, axis_rewards, axis_cumulative_rewards, environment, path_of_episode):
    global rewards, steps_episode  # Referencing global variables
    final_episode = agent.rewards - 1  # Check if it's the last episode
    if episode_number == final_episode:  # Update plots only after the last training episode
        render_qvalue_heatmap(agent.q_table, axis_qvalues)
        display_agent_policy_on_grid(agent.q_table, axis_policy)
        axis_policy.set_title(r'$\bf{Visualization\ of\ Policy\ Grades}$', fontsize=14)
        axis_qvalues.set_title(r'$\bf{Heatmap\ of\ Q-values}$', fontsize=14)
        axis_path.set_title(r'$\bf{Path\ Taken\ by\ Agent}$', fontsize=14)

        # Plot the number of steps per episode
        axis_steps.plot(range(episode_number + 1), steps_episode, color='blue')
        axis_steps.set_xlabel('Episode')
        axis_steps.set_ylabel('Steps')
        axis_steps.set_title(r'$\bf{Episode\ vs.\ Steps}$', fontsize=14)

        # Plot the rewards obtained per episode
        axis_rewards.plot(range(episode_number + 1), rewards, color='magenta')
        axis_rewards.set_xlabel('Episode')
        axis_rewards.set_ylabel('Reward')
        axis_rewards.set_title(r'$\bf{Episode\ vs.\ Rewards}$', fontsize=14)

        # Plot the accumulated rewards over all episodes
        accumulated_rewards = np.cumsum(rewards)
        axis_cumulative_rewards.plot(range(episode_number + 1), accumulated_rewards, color='green')
        axis_cumulative_rewards.set_xlabel('Episode')
        axis_cumulative_rewards.set_ylabel('Total Reward')
        axis_cumulative_rewards.set_title(r'$\bf{Episode\ vs.\ Cumulative\ Reward}$', fontsize=14)

        # Display the agent's path during the episode
        draw_agent_route(environment.grid_size, path_of_episode, axis_path)

        # Adjust the spacing between plot rows and columns
        plt.subplots_adjust(horizontal_space=0.3, vertical_space=0.4)