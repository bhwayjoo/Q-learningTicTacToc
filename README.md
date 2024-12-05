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

## Webography

### Reinforcement Learning Resources
1. [Q-Learning Introduction - Towards Data Science](https://towardsdatascience.com/q-learning-for-beginners-d5757334386c)
   - Comprehensive introduction to Q-Learning concepts
   - Basic implementation examples

2. [OpenAI Spinning Up - Key Papers in Deep RL](https://spinningup.openai.com/en/latest/spinningup/keypapers.html)
   - Collection of important reinforcement learning papers
   - Theoretical foundations of Q-Learning

3. [Sutton & Barto - Reinforcement Learning Book](http://incompleteideas.net/book/the-book-2nd.html)
   - Classic textbook on reinforcement learning
   - Chapter 6: Temporal-Difference Learning

### Python Implementation Resources
4. [Tkinter Documentation - Python](https://docs.python.org/3/library/tkinter.html)
   - Official Python Tkinter documentation
   - GUI implementation guidelines

5. [NumPy Documentation](https://numpy.org/doc/stable/)
   - Array operations and mathematical functions
   - Used for Q-table operations

6. [Matplotlib Visualization Guide](https://matplotlib.org/stable/tutorials/index.html)
   - Plotting and visualization tutorials
   - Used for learning progress visualization

### Game Development Resources
7. [Tic Tac Toe Game Theory](https://en.wikipedia.org/wiki/Tic-tac-toe#Strategy)
   - Game strategy and optimal play
   - State space complexity analysis

8. [Game Programming Patterns](http://gameprogrammingpatterns.com/)
   - Software design patterns for games
   - State management techniques

### Additional Learning Materials
9. [Deep Reinforcement Learning Course](https://huggingface.co/deep-rl-course/unit0/introduction)
   - Advanced RL concepts and implementations
   - Modern approaches to game AI

10. [Stanford CS234: Reinforcement Learning](https://web.stanford.edu/class/cs234/)
    - Academic course materials
    - Theoretical foundations

### Code Examples and Tutorials
11. [GitHub - Q-Learning Examples](https://github.com/topics/q-learning)
    - Open source implementations
    - Community contributions

12. [Real Python - Tkinter Tutorials](https://realpython.com/python-gui-tkinter/)
    - GUI development best practices
    - Event handling examples

### Development Tools
13. [Python Type Hints Documentation](https://docs.python.org/3/library/typing.html)
    - Type annotation guidelines
    - Code maintainability

14. [Pickle Documentation](https://docs.python.org/3/library/pickle.html)
    - Object serialization
    - Model saving/loading implementation

### Best Practices and Standards
15. [Python Style Guide (PEP 8)](https://www.python.org/dev/peps/pep-0008/)
    - Code formatting standards
    - Naming conventions

16. [Clean Code in Python](https://github.com/zedr/clean-code-python)
    - Python-specific clean code principles
    - Code organization guidelines

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.
