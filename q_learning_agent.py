import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional
import os
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95, epsilon: float = 0.1):
        self.q_table: Dict[str, Dict[int, float]] = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.training_history: List[Dict[str, float]] = []
        self.wins = 0
        self.games_played = 0

    def get_state_key(self, board: List[str]) -> str:
        """Convert board state to string key"""
        return ''.join(x if x != "" else "-" for x in board)

    def get_valid_actions(self, board: List[str]) -> List[int]:
        """Get list of valid moves"""
        return [i for i, x in enumerate(board) if x == ""]

    def get_q_values(self, state_key: str) -> Dict[int, float]:
        """Get Q-values for a state, initializing if needed"""
        if state_key not in self.q_table:
            self.q_table[state_key] = {i: 0.0 for i in range(9)}
        return self.q_table[state_key]

    def choose_action(self, board: List[str]) -> Optional[int]:
        """Choose action using epsilon-greedy strategy"""
        valid_actions = self.get_valid_actions(board)
        if not valid_actions:
            return None

        state_key = self.get_state_key(board)
        q_values = self.get_q_values(state_key)

        # Exploration
        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)

        # Exploitation
        valid_q_values = {action: q_values[action] for action in valid_actions}
        max_q = max(valid_q_values.values())
        best_actions = [action for action, q in valid_q_values.items() if q == max_q]
        return np.random.choice(best_actions)

    def learn(self, state: List[str], action: int, reward: float, next_state: List[str], terminal: bool):
        """Update Q-values using Q-learning algorithm"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        # Get current Q-value
        current_q = self.get_q_values(state_key)[action]

        # Calculate new Q-value
        if terminal:
            target_q = reward
        else:
            next_q_values = self.get_q_values(next_state_key)
            next_valid_actions = self.get_valid_actions(next_state)
            if next_valid_actions:
                max_next_q = max(next_q_values[a] for a in next_valid_actions)
                target_q = reward + self.discount_factor * max_next_q
            else:
                target_q = reward

        # Update Q-value
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[state_key][action] = new_q

        # Update training history
        self.training_history.append({
            'q_value': new_q,
            'reward': reward,
            'terminal': terminal
        })
        
        if terminal:
            self.games_played += 1
            if reward > 0:
                self.wins += 1

    def plot_training_history(self, window_size: int = 100):
        """Plot training history with moving averages"""
        if not self.training_history:
            return
            
        # Extract data from training history
        q_values = [entry['q_value'] for entry in self.training_history]
        rewards = [entry['reward'] for entry in self.training_history]
        
        # Calculate moving averages
        def moving_average(data: List[float], window: int) -> np.ndarray:
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        if len(q_values) > window_size:
            x_axis = range(len(q_values) - window_size + 1)
            q_values_ma = moving_average(q_values, window_size)
            rewards_ma = moving_average(rewards, window_size)
            
            plt.figure(figsize=(12, 6))
            
            # Plot Q-values
            plt.subplot(2, 1, 1)
            plt.plot(x_axis, q_values_ma, label='Q-values')
            plt.title(f'Average Q-values (Window Size: {window_size})')
            plt.xlabel('Training Step')
            plt.ylabel('Q-value')
            plt.legend()
            
            # Plot rewards
            plt.subplot(2, 1, 2)
            plt.plot(x_axis, rewards_ma, label='Rewards', color='orange')
            plt.title(f'Average Rewards (Window Size: {window_size})')
            plt.xlabel('Training Step')
            plt.ylabel('Reward')
            plt.legend()
            
            plt.tight_layout()
            plt.show()

    def get_win_rate(self) -> float:
        """Calculate win rate"""
        return self.wins / max(1, self.games_played)

    def save_model(self, filepath: str):
        """Save Q-table and training history to file"""
        model_data = {
            'q_table': self.q_table,
            'training_history': self.training_history,
            'wins': self.wins,
            'games_played': self.games_played,
            'params': {
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon': self.epsilon
            }
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath: str):
        """Load Q-table and training history from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            self.q_table = model_data['q_table']
            self.training_history = model_data['training_history']
            self.wins = model_data['wins']
            self.games_played = model_data['games_played']
            params = model_data['params']
            self.learning_rate = params['learning_rate']
            self.discount_factor = params['discount_factor']
            self.epsilon = params['epsilon']

    def reset_stats(self):
        """Reset training statistics"""
        self.wins = 0
        self.games_played = 0
        self.training_history = []
