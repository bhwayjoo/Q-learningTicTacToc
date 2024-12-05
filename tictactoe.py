import tkinter as tk
from tkinter import messagebox, ttk
import random
import time
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import os
from q_learning_agent import QLearningAgent

class TicTacToe:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Tic Tac Toe with Q-Learning")
        
        # Initialize game state
        self.board = [""] * 9
        self.current_player = "X"
        self.game_over = False
        
        # Initialize AI agents
        self.agent_x = QLearningAgent(epsilon=0.2)
        self.agent_o = QLearningAgent(epsilon=0.2)
        
        # Load saved models if they exist
        if os.path.exists("models/agent_x.pkl"):
            self.agent_x.load_model("models/agent_x.pkl")
        if os.path.exists("models/agent_o.pkl"):
            self.agent_o.load_model("models/agent_o.pkl")
        
        # Game statistics
        self.stats = {"X": 0, "O": 0, "Tie": 0}
        
        # Game settings
        self.ai_mode = True
        self.self_play_mode = False
        self.game_speed = tk.DoubleVar(value=0.5)
        self.training_mode = False
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the game user interface"""
        self.main_frame = tk.Frame(self.window, bg='#f0f0f0')
        self.main_frame.pack(padx=10, pady=10)
        
        # Setup UI components
        self.setup_game_modes()
        self.setup_game_board()
        self.setup_controls()
        self.setup_stats()
        self.setup_speed_control()  # Add speed control setup
        
        # Initialize buttons list
        self.buttons = []
        
        # Create game board buttons
        for i in range(3):
            for j in range(3):
                button = tk.Button(
                    self.board_frame,
                    text="",
                    font=('Arial', 20),
                    width=5,
                    height=2,
                    command=lambda row=i, col=j: self.button_click(row, col)
                )
                button.grid(row=i, column=j, padx=2, pady=2)
                self.buttons.append(button)

    def setup_game_modes(self):
        """Setup game mode controls"""
        self.mode_frame = tk.Frame(self.main_frame, bg='#f0f0f0')
        self.mode_frame.pack(pady=5)
        
        # Game mode selection
        modes_frame = tk.LabelFrame(self.mode_frame, text="Game Mode", bg='#f0f0f0')
        modes_frame.pack(pady=5)
        
        self.game_mode = tk.StringVar(value="ai")
        tk.Radiobutton(modes_frame, text="User vs AI", variable=self.game_mode, 
                      value="ai", command=self.change_game_mode, bg='#f0f0f0').pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(modes_frame, text="User vs User", variable=self.game_mode, 
                      value="human", command=self.change_game_mode, bg='#f0f0f0').pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(modes_frame, text="AI vs AI", variable=self.game_mode, 
                      value="self_play", command=self.change_game_mode, bg='#f0f0f0').pack(side=tk.LEFT, padx=5)
        
        # Training controls
        training_frame = tk.LabelFrame(self.mode_frame, text="Training Settings", bg='#f0f0f0')
        training_frame.pack(pady=5)
        
        # Number of training games
        self.games_var = tk.StringVar(value="1000")
        tk.Label(training_frame, text="Training Games:", bg='#f0f0f0').pack(side=tk.LEFT, padx=5)
        tk.Entry(training_frame, textvariable=self.games_var, width=8).pack(side=tk.LEFT, padx=5)
        
        # Training button
        self.train_button = tk.Button(training_frame, text="Start Training", 
                                    command=self.start_training, bg='#2196F3', fg='white')
        self.train_button.pack(side=tk.LEFT, padx=5)
        
        # Training progress
        self.progress_var = tk.StringVar(value="")
        self.progress_label = tk.Label(self.mode_frame, textvariable=self.progress_var, 
                                     bg='#f0f0f0', fg='#666')
        self.progress_label.pack(pady=2)

    def setup_game_board(self):
        """Setup the game board"""
        self.board_frame = tk.Frame(self.main_frame, bg='#f0f0f0')
        self.board_frame.pack(pady=10)

    def setup_controls(self):
        """Setup control buttons"""
        self.control_frame = tk.Frame(self.main_frame, bg='#f0f0f0')
        self.control_frame.pack(pady=10)
        
        # Reset button
        reset_button = tk.Button(
            self.control_frame, 
            text="Reset Game", 
            font=('Arial', 12), 
            command=self.reset_game,
            bg='#4CAF50',
            fg='white'
        )
        reset_button.pack(side=tk.LEFT, padx=5)
        
        # Save button
        save_button = tk.Button(
            self.control_frame, 
            text="Save AI", 
            font=('Arial', 12), 
            command=self.save_models,
            bg='#2196F3',
            fg='white'
        )
        save_button.pack(side=tk.LEFT, padx=5)
        
        # Load button
        load_button = tk.Button(
            self.control_frame, 
            text="Load AI", 
            font=('Arial', 12), 
            command=self.load_models,
            bg='#FFC107',
            fg='white'
        )
        load_button.pack(side=tk.LEFT, padx=5)
        
        # Show Stats button
        stats_button = tk.Button(
            self.control_frame, 
            text="Show Stats", 
            font=('Arial', 12), 
            command=self.show_stats,
            bg='#9C27B0',
            fg='white'
        )
        stats_button.pack(side=tk.LEFT, padx=5)
        
    def setup_stats(self):
        """Setup statistics display"""
        self.stats_frame = tk.Frame(self.main_frame, bg='#f0f0f0')
        self.stats_frame.pack(pady=10)
        
        self.stats_label = tk.Label(
            self.stats_frame, 
            text="Stats - X: 0  O: 0  Ties: 0",
            font=('Arial', 12), 
            bg='#f0f0f0'
        )
        self.stats_label.pack()
        
        self.turn_label = tk.Label(
            self.main_frame, 
            text="Player X's turn", 
            font=('Arial', 12), 
            bg='#f0f0f0'
        )
        self.turn_label.pack(pady=5)
        
    def setup_speed_control(self):
        """Setup AI move speed control"""
        speed_frame = tk.Frame(self.main_frame, bg='#f0f0f0')
        speed_frame.pack(pady=5)
        
        tk.Label(speed_frame, text="AI Move Speed:", bg='#f0f0f0').pack(side=tk.LEFT, padx=5)
        
        speed_scale = ttk.Scale(
            speed_frame,
            from_=0.1,
            to=2.0,
            variable=self.game_speed,
            orient=tk.HORIZONTAL,
            length=200
        )
        speed_scale.pack(side=tk.LEFT, padx=5)

    def change_game_mode(self):
        """Handle game mode changes"""
        mode = self.game_mode.get()
        self.ai_mode = mode == "ai"
        self.self_play_mode = mode == "self_play"
        self.reset_game()
        
        # Update UI based on mode
        if mode == "self_play":
            self.progress_var.set("AI vs AI mode: Watch two AI agents play against each other")
            self.window.after(1000, self.play_ai_vs_ai_game)
        elif mode == "ai":
            self.progress_var.set("User vs AI mode: You play as X, AI plays as O")
        else:
            self.progress_var.set("User vs User mode: Two players take turns")

    def start_training(self):
        """Start the AI training process"""
        try:
            num_games = int(self.games_var.get())
            if num_games <= 0:
                raise ValueError("Number of games must be positive")
            
            # Disable controls during training
            self.train_button.config(state=tk.DISABLED)
            for button in self.buttons:
                button.config(state=tk.DISABLED)
            
            # Reset agents' training history
            self.agent_x.reset_stats()
            self.agent_o.reset_stats()
            
            # Start training
            self.training_mode = True
            self.window.after(100, lambda: self.run_training(num_games))
            
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            self.train_button.config(state=tk.NORMAL)
            for button in self.buttons:
                button.config(state=tk.NORMAL)
            self.training_mode = False

    def run_training(self, num_games: int):
        """Execute the training process"""
        try:
            # Save original states
            original_mode = self.game_mode.get()
            original_epsilon_x = self.agent_x.epsilon
            original_epsilon_o = self.agent_o.epsilon
            
            # Set training mode
            self.training_mode = True
            
            # Set training parameters
            self.agent_x.epsilon = 0.3  # Increase exploration during training
            self.agent_o.epsilon = 0.3
            
            wins_x = 0
            wins_o = 0
            ties = 0
            
            for game in range(num_games):
                self.reset_game(update_display=False)
                game_history = []
                
                while True:
                    current_agent = self.agent_x if self.current_player == "X" else self.agent_o
                    current_state = self.board.copy()
                    
                    action = current_agent.choose_action(self.board)
                    if action is None:
                        break
                    
                    game_history.append((current_state.copy(), action, self.current_player))
                    game_over = self.make_move(action)
                    
                    # Update progress every 10 games
                    if game % 10 == 0:
                        self.progress_var.set(f"Training Progress: {game}/{num_games} games\n"
                                           f"X: {wins_x}, O: {wins_o}, Ties: {ties}")
                        self.window.update()
                    
                    if game_over:
                        winner = self.check_winner()
                        if winner == "X":
                            wins_x += 1
                        elif winner == "O":
                            wins_o += 1
                        else:
                            ties += 1
                        
                        # Update Q-values for all moves in the game
                        reward = self.get_reward(winner if winner else "Tie")
                        for state, move, player in reversed(game_history):
                            agent = self.agent_x if player == "X" else self.agent_o
                            agent.learn(state, move, reward, self.board, True)
                            reward = -reward  # Flip reward for opponent
                        break
            
            # Training complete
            self.progress_var.set(f"Training Complete!\nFinal Stats - X: {wins_x}, O: {wins_o}, Ties: {ties}")
            
            # Restore original states
            self.game_mode.set(original_mode)
            self.agent_x.epsilon = original_epsilon_x
            self.agent_o.epsilon = original_epsilon_o
            self.training_mode = False
            
            # Re-enable controls
            self.train_button.config(state=tk.NORMAL)
            for button in self.buttons:
                button.config(state=tk.NORMAL)
            
            # Show training results
            self.plot_learning_progress()
            
        except Exception as e:
            messagebox.showerror("Training Error", f"An error occurred during training: {str(e)}")
            self.train_button.config(state=tk.NORMAL)
            for button in self.buttons:
                button.config(state=tk.NORMAL)

    def plot_learning_progress(self):
        """Plot learning progress for both agents"""
        try:
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
            
            # Plot Agent X's learning curves
            if self.agent_x.training_history:
                q_values_x = [entry['q_value'] for entry in self.agent_x.training_history]
                rewards_x = [entry['reward'] for entry in self.agent_x.training_history]
                steps_x = range(len(q_values_x))
                
                ax1.plot(steps_x, q_values_x, label='Q-values', color='blue', alpha=0.6)
                ax1.plot(steps_x, rewards_x, label='Rewards', color='green', alpha=0.6)
                ax1.set_title("Agent X Learning Progress")
                ax1.set_xlabel("Training Steps")
                ax1.set_ylabel("Value")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # Plot Agent O's learning curves
            if self.agent_o.training_history:
                q_values_o = [entry['q_value'] for entry in self.agent_o.training_history]
                rewards_o = [entry['reward'] for entry in self.agent_o.training_history]
                steps_o = range(len(q_values_o))
                
                ax2.plot(steps_o, q_values_o, label='Q-values', color='red', alpha=0.6)
                ax2.plot(steps_o, rewards_o, label='Rewards', color='orange', alpha=0.6)
                ax2.set_title("Agent O Learning Progress")
                ax2.set_xlabel("Training Steps")
                ax2.set_ylabel("Value")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Visualization Error", f"Failed to plot learning progress: {str(e)}")

    def check_winner(self) -> Optional[str]:
        """Check for a winner in the game"""
        # Check rows
        for i in range(0, 9, 3):
            if self.board[i] == self.board[i+1] == self.board[i+2] != "":
                return self.board[i]
        
        # Check columns
        for i in range(3):
            if self.board[i] == self.board[i+3] == self.board[i+6] != "":
                return self.board[i]
        
        # Check diagonals
        if self.board[0] == self.board[4] == self.board[8] != "":
            return self.board[0]
        if self.board[2] == self.board[4] == self.board[6] != "":
            return self.board[2]
        
        return None

    def is_board_full(self) -> bool:
        """Check if the board is full"""
        return "" not in self.board

    def get_reward(self, result: str) -> float:
        """Calculate reward based on game result"""
        if result == "X":
            return 1.0
        elif result == "O":
            return -1.0
        else:  # Tie
            return 0.0

    def make_move(self, position: int) -> bool:
        """Make a move and update the game state"""
        if self.board[position] == "":
            # Update board and UI
            self.board[position] = self.current_player
            self.buttons[position].config(
                text=self.current_player,
                fg='blue' if self.current_player == 'X' else 'red'
            )
            
            # Check for game end
            winner = self.check_winner()
            if winner:
                self.stats[winner] += 1
                self.update_stats_display()
                # Only show game over message in human vs human or human vs AI mode
                if not self.self_play_mode and not self.training_mode:
                    messagebox.showinfo("Game Over", f"Player {winner} wins!")
                return True
                
            if "" not in self.board:
                self.stats["Tie"] += 1
                self.update_stats_display()
                # Only show game over message in human vs human or human vs AI mode
                if not self.self_play_mode and not self.training_mode:
                    messagebox.showinfo("Game Over", "It's a tie!")
                return True
            
            # Switch players
            self.current_player = "O" if self.current_player == "X" else "X"
            self.turn_label.config(text=f"Player {self.current_player}'s turn")
            
            return False
            
        return False

    def play_ai_vs_ai_game(self):
        """Execute AI vs AI gameplay"""
        if not self.self_play_mode:
            return
            
        current_agent = self.agent_x if self.current_player == "X" else self.agent_o
        action = current_agent.choose_action(self.board)
        
        if action is not None and self.board[action] == "":
            current_state = self.board.copy()
            game_over = self.make_move(action)
            
            # Update Q-values
            if not game_over:
                reward = self.get_reward(self.current_player)
                current_agent.learn(current_state, action, reward, self.board, False)
                self.window.after(int(self.game_speed.get() * 1000), self.play_ai_vs_ai_game)
            else:
                # Game ended, update Q-values with final reward
                winner = self.check_winner()
                reward = self.get_reward(winner if winner else "")
                current_agent.learn(current_state, action, reward, self.board, True)
                
                # Start new game after delay
                self.window.after(2000, self.start_new_ai_game)

    def start_new_ai_game(self):
        """Start a new AI vs AI game"""
        if self.self_play_mode:
            self.reset_game()
            self.window.after(1000, self.play_ai_vs_ai_game)

    def button_click(self, row: int, col: int):
        """Handle button click events"""
        if self.self_play_mode:
            return
            
        position = row * 3 + col
        if self.board[position] == "":
            if self.game_mode.get() == "human" or (self.game_mode.get() == "ai" and self.current_player == "X"):
                current_state = self.board.copy()
                game_over = self.make_move(position)
                
                if not game_over and self.game_mode.get() == "ai":
                    # Add a small delay before AI move
                    self.window.after(int(self.game_speed.get() * 1000), self.make_ai_move)

    def make_ai_move(self):
        """Make a move for the AI player"""
        if not self.ai_mode or self.self_play_mode or self.current_player == "X":
            return
            
        current_agent = self.agent_o  # AI always plays as O in User vs AI mode
        action = current_agent.choose_action(self.board)
        
        if action is not None and self.board[action] == "":
            current_state = self.board.copy()
            game_over = self.make_move(action)
            
            if not game_over:
                # Update Q-values for non-terminal states
                reward = self.get_reward(None)
                current_agent.learn(current_state, action, reward, self.board, False)
            else:
                # Update Q-values for terminal states
                winner = self.check_winner()
                reward = self.get_reward(winner if winner else "Tie")
                current_agent.learn(current_state, action, reward, self.board, True)

    def reset_game(self, update_display=True):
        """Reset the game state"""
        self.board = [""] * 9
        self.current_player = "X"
        
        if update_display:
            for button in self.buttons:
                button.config(text="", state=tk.NORMAL)
            self.turn_label.config(text="Player X's turn")
            
            # If in AI vs AI mode, start new game
            if self.self_play_mode:
                self.window.after(1000, self.play_ai_vs_ai_game)

    def update_stats_display(self):
        """Update the statistics display"""
        self.stats_label.config(
            text=f"Stats - X: {self.stats['X']}  O: {self.stats['O']}  Ties: {self.stats['Tie']}"
        )
        self.turn_label.config(text=f"Player {self.current_player}'s turn")

    def train_agents(self, num_games: int):
        """Train the AI agents through self-play"""
        original_mode = self.self_play_mode
        self.self_play_mode = True
        
        # Increase exploration during training
        original_epsilon_x = self.agent_x.epsilon
        original_epsilon_o = self.agent_o.epsilon
        self.agent_x.epsilon = 0.3
        self.agent_o.epsilon = 0.3
        
        for game in range(num_games):
            self.reset_game()
            while True:
                current_agent = self.agent_x if self.current_player == "X" else self.agent_o
                current_state = self.board.copy()
                
                action = current_agent.choose_action(self.board)
                if action is None:
                    break
                    
                game_over = self.make_move(action)
                reward = self.get_reward(self.current_player)
                
                # Update Q-values
                current_agent.learn(
                    current_state,
                    action,
                    reward,
                    self.board,
                    game_over
                )
                
                if game_over:
                    break
            
            # Update progress
            if (game + 1) % 100 == 0:
                self.turn_label.config(text=f"Training... {game + 1}/{num_games} games")
                self.window.update()
        
        # Restore original settings
        self.self_play_mode = original_mode
        self.agent_x.epsilon = original_epsilon_x
        self.agent_o.epsilon = original_epsilon_o
        self.turn_label.config(text="Training complete!")

    def show_stats(self):
        """Display game statistics"""
        try:
            plt.figure(figsize=(8, 6))
            stats_data = [self.stats['X'], self.stats['O'], self.stats['Tie']]
            plt.bar(['X Wins', 'O Wins', 'Ties'], stats_data)
            plt.title('Game Statistics')
            plt.ylabel('Number of Games')
            plt.show()
        except Exception as e:
            messagebox.showerror("Statistics Error", f"Failed to show statistics: {str(e)}")

    def save_models(self):
        """Save AI models"""
        try:
            models_dir = os.path.join(os.path.dirname(__file__), 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            agent_x_path = os.path.join(models_dir, 'agent_x.pkl')
            agent_o_path = os.path.join(models_dir, 'agent_o.pkl')
            
            self.agent_x.save_model(agent_x_path)
            self.agent_o.save_model(agent_o_path)
            messagebox.showinfo("Success", "AI models saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save AI models: {str(e)}")

    def load_models(self):
        """Load AI models"""
        try:
            models_dir = os.path.join(os.path.dirname(__file__), 'models')
            agent_x_path = os.path.join(models_dir, 'agent_x.pkl')
            agent_o_path = os.path.join(models_dir, 'agent_o.pkl')
            
            if not os.path.exists(agent_x_path) or not os.path.exists(agent_o_path):
                raise FileNotFoundError("Model files not found. Train the AI first.")
            
            self.agent_x.load_model(agent_x_path)
            self.agent_o.load_model(agent_o_path)
            messagebox.showinfo("Success", "AI models loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load AI models: {str(e)}")

    def run(self):
        """Start the game"""
        self.window.mainloop()

if __name__ == "__main__":
    game = TicTacToe()
    game.run()
