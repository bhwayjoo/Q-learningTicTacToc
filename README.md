# Tic Tac Toe with Q-Learning AI

An intelligent Tic Tac Toe game implementation using Q-Learning reinforcement learning algorithm. The AI agents learn and improve their gameplay through training.

## Features

- Multiple game modes:
  - User vs AI
  - User vs User
  - AI vs AI (self-play)
- Interactive training mode
- Learning visualization
- Game statistics tracking
- Save/Load AI models
- Adjustable AI move speed

## Requirements

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository or download the source code
2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the game:
```bash
python tictactoe.py
```

2. Game Controls:
- Select game mode using radio buttons
- Click board cells to make moves
- Use "Reset Game" to start a new game
- Use "Save AI" to save trained models
- Use "Load AI" to load previously saved models
- Use "Show Stats" to view game statistics

3. Training the AI:
- Select "AI vs AI" mode
- Enter number of training games
- Click "Start Training"
- Watch the training progress
- View learning curves after training

## Project Structure

- `tictactoe.py`: Main game logic and GUI
- `q_learning_agent.py`: Q-Learning AI implementation
- `requirements.txt`: Project dependencies

## How It Works

The AI agents use Q-Learning, a reinforcement learning algorithm, to learn optimal moves:
- States: Board configurations
- Actions: Available moves (empty cells)
- Rewards: Win (+1.0), Loss (-1.0), Draw (0.0)
- Learning parameters:
  - Learning rate: How much to update Q-values
  - Discount factor: Value of future rewards
  - Epsilon: Exploration vs exploitation trade-off

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.
