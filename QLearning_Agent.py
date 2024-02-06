
import numpy as np

class QLearning_Agent:
    # A Q-table for storage. Q-values representing the expected rewards for each possible action in every state
    def __init__(self, state_size, number_actions, rate_learning, factor_discount, rate_exploration):
        """
        Construct a Q-Learning Agent.

        Args:
        - state_size (tuple): Size of the state space.
        - number_actions (int): Count of potential actions.
        - rate_learning (float): Rate at which the agent learns (alpha).
        - factor_discount (float): Rate at which future rewards are discounted (gamma).
        - rate_exploration (float): Probability of choosing a random action (epsilon).
        """
        # Set up a table for Q-values with initial zero values
        self.qtable = np.zeros(state_size + (number_actions,))
        self.rate_learning = rate_learning
        self.factor_discount = factor_discount
        self.rate_exploration = rate_exploration
        self.number_actions = number_actions
        self.path_episode = [] # Record of the agent's trajectory in an episode


    # Using the epsilon-greedy approach to implement action selection logic
    def action_choose(self, state):
        """
        Decide on an action based on the current state using an epsilon-greedy approach.

        Args:
        - state (tuple): The current state from which to choose an action.

        Returns:
        - (int): The selected action.
        """
        if np.random.rand() < self.rate_exploration:
            return np.random.randint(self.number_actions)
        else:
            return np.argmax(self.qtable[state])
        
    
    # Using the Q-learning update rule to set up Q-table update code
    def qtable_update(self, state, action, reward, state_next):
        """
        Update the Q-table with a new value based on the agent's experience.

        Args:
        - state (tuple): The state prior to taking the action.
        - action (int): The action that was taken.
        - reward (float): The reward received after taking the action.
        - state_next (tuple): The state after the action was taken.
        """

        q_value_current = self.qtable[state][action]
        q_value_next_max = np.max(self.qtable[state_next])
        q_value_updated = (1 - self.rate_learning) * q_value_current + \
                          self.rate_learning * (reward + self.factor_discount * q_value_next_max)
        self.qtable[state][action] = q_value_updated

    def episode_path_reset(self):
        """
        Clear the record of the agent's path for a new episode.
        """
        self.path_episode = []

    def get_policy_optimal(self, obstacles):
        """
        Create an optimal policy based on the highest Q-values in the Q-table.

        Args:
        - obstacles (np.array): A grid indicating where obstacles are present.

        Returns:
        - policy_optimal (np.array): The derived optimal policy grid.
        """
        # Initialize the policy grid, defaulting to -1 for obstacles
        policy_optimal = np.full(self.qtable.shape[:-1], -1, dtype=int) 

        for state in np.ndindex(policy_optimal.shape):
            if not obstacles[state]:
                q_value_max = np.max(self.qtable[state])
                actions_best = np.where(self.qtable[state] == q_value_max)[0]
                action_chosen = np.random.choice(actions_best)  # Select among the best actions
                policy_optimal[state] = action_chosen

        return policy_optimal