# Chess Reinforcement Learning with Visualization

A visual chess game with integrated reinforcement learning capabilities. Watch the AI learn to play chess in real-time!

## Features
- Interactive chess board visualization
- Reinforcement Learning agent training
- Real-time learning metrics and visualization
- Play against the AI or watch it train

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Train the RL agent
python main.py --mode train

# Play against the AI
python main.py --mode play

# Watch two AI agents play
python main.py --mode watch
```

## Components
- `chess_board.py`: Visual chess board using pygame
- `rl_agent.py`: Deep Q-Network agent for chess
- `training.py`: Training loop with visualization
- `main.py`: Main entry point
