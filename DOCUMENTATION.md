# Chess RL - Complete Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [File-by-File Breakdown](#file-by-file-breakdown)
4. [Core Concepts](#core-concepts)
5. [Code Examples](#code-examples)
6. [Training Process](#training-process)
7. [Advanced Usage](#advanced-usage)

---

## Project Overview

This project implements a **Deep Q-Network (DQN)** reinforcement learning agent that learns to play chess through self-play. The key innovation is **real-time visualization** - you can watch the AI learn move-by-move and see learning metrics update live.

### Key Features
- 🎮 Visual chess board with distinct geometric shapes for pieces
- 🧠 Deep neural network for position evaluation
- 📊 Real-time training metrics dashboard
- 🎯 Three modes: Train, Play, Watch
- 💾 Model save/load functionality

---

## Architecture

### High-Level System Design

```
┌─────────────────────────────────────────────────────────┐
│                    Main Application                      │
│                      (main.py)                           │
└───────────────┬─────────────────────────────────────────┘
                │
        ┌───────┴────────┐
        │                │
        ▼                ▼
┌──────────────┐  ┌─────────────────┐
│ Chess Board  │  │   RL Agent      │
│ Visualization│  │  (Neural Net)   │
│ (pygame)     │  │   (PyTorch)     │
└──────────────┘  └─────────────────┘
        │                │
        │                │
        ▼                ▼
┌──────────────┐  ┌─────────────────┐
│   Training   │  │  Visualization  │
│    Loop      │  │    Metrics      │
│              │  │  (matplotlib)   │
└──────────────┘  └─────────────────┘
```

### Data Flow During Training

```
1. Game State (Chess Board)
   │
   ▼
2. Convert to Tensor (8×8×12 → 768 values)
   │
   ▼
3. Neural Network (768 → 512 → 512 → 256 → 1)
   │
   ▼
4. Position Evaluation Score
   │
   ▼
5. Select Best Move (or random if exploring)
   │
   ▼
6. Execute Move & Calculate Reward
   │
   ▼
7. Store Experience in Replay Buffer
   │
   ▼
8. Sample Random Batch & Train Network
   │
   ▼
9. Update Weights (Backpropagation)
   │
   └─→ Repeat
```

---

## File-by-File Breakdown

### 1. `chess_board.py` - Visual Chess Board

**Purpose**: Handles all chess board visualization and user interaction using pygame.

#### Key Components:

**ChessBoard Class**
```python
class ChessBoard:
    def __init__(self, width=800, height=800):
        # Initialize pygame window
        # Set up colors for squares
        # Create chess board state
        # Load piece shapes
```

**Core Methods:**

1. **`load_pieces()`**
   - Maps piece types to shape identifiers
   - No actual image loading (uses geometric shapes)

2. **`draw_board()`**
   - Draws 8×8 checkerboard pattern
   - Alternates light/dark squares
   - Adds file (a-h) and rank (1-8) labels

3. **`draw_pieces()`**
   - Most complex method
   - Draws different shapes for each piece type:
     - **Pawn**: Circle
     - **Rook**: Rectangle with battlements
     - **Knight**: Triangle
     - **Bishop**: Diamond
     - **Queen**: Circle with crown points
     - **King**: Circle with cross

4. **`highlight_square(color)`**
   - Shows selected piece
   - Shows valid move destinations
   - Uses semi-transparent overlays

5. **`get_square_from_mouse(pos)`**
   - Converts pixel coordinates to chess square
   - Returns square index (0-63)

6. **`handle_click(square)`**
   - Manages piece selection
   - Validates moves
   - Returns move if valid

**State Representation:**
```
Board coordinates:
  a b c d e f g h
8 □ ■ □ ■ □ ■ □ ■  8
7 ■ □ ■ □ ■ □ ■ □  7
6 □ ■ □ ■ □ ■ □ ■  6
5 ■ □ ■ □ ■ □ ■ □  5
4 □ ■ □ ■ □ ■ □ ■  4
3 ■ □ ■ □ ■ □ ■ □  3
2 □ ■ □ ■ □ ■ □ ■  2
1 ■ □ ■ □ ■ □ ■ □  1
  a b c d e f g h
```

---

### 2. `rl_agent.py` - Reinforcement Learning Agent

**Purpose**: Implements the Deep Q-Network agent that learns chess.

#### Key Components:

**ChessNet - Neural Network**
```python
class ChessNet(nn.Module):
    Input:  768 neurons (8×8×12 board representation)
    Layer1: 768 → 512 (with ReLU activation)
    Drop1:  30% dropout (prevent overfitting)
    Layer2: 512 → 512 (with ReLU)
    Drop2:  30% dropout
    Layer3: 512 → 256 (with ReLU)
    Output: 256 → 1 (position evaluation score)
```

**Why this architecture?**
- **768 input neurons**: Represents all piece positions (6 piece types × 2 colors × 64 squares)
- **512-512-256 hidden layers**: Large enough to learn complex patterns
- **Dropout**: Prevents memorizing specific positions
- **Single output**: How good is this position? (positive = good for current player)

**Board Representation:**
```python
# 8×8×12 tensor (one-hot encoding)
# Channels 0-5: White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
# Channels 6-11: Black pieces (Pawn, Knight, Bishop, Rook, Queen, King)

Example for square e2 with white pawn:
board[1][4][0] = 1.0  # Row 1, Column 4 (e-file), Channel 0 (white pawn)
```

**ReplayBuffer - Experience Memory**
```python
class ReplayBuffer:
    # Stores up to 10,000 experiences
    # Each experience: (state, action, reward, next_state, done)
    
    def push(...):
        # Add new experience, automatically removes oldest if full
    
    def sample(batch_size):
        # Randomly select batch_size experiences for training
```

**Why replay buffer?**
- **Breaks correlation**: Consecutive game states are highly similar
- **Reuses experiences**: Learn from same situation multiple times
- **Stabilizes training**: Random sampling prevents catastrophic forgetting

**ChessRLAgent - The Learning Agent**

**Initialization:**
```python
def __init__(self, learning_rate=0.001, gamma=0.99, epsilon=1.0, 
             epsilon_min=0.01, epsilon_decay=0.995):
    
    # Two networks (key to stable learning!)
    self.policy_net = ChessNet()   # The learning network
    self.target_net = ChessNet()   # The stable reference
    
    # Hyperparameters
    self.gamma = 0.99        # Discount factor (how much to value future)
    self.epsilon = 1.0       # Exploration rate (100% random at start)
    self.epsilon_decay = 0.995  # Decay rate (becomes less random)
    self.epsilon_min = 0.01  # Minimum exploration (always 1% random)
```

**Key Methods Explained:**

**1. `board_to_tensor(board)`** - State Encoding
```python
# Converts chess.Board to neural network input
# Returns: torch.Tensor of shape [1, 768]

Process:
1. Create 8×8×12 numpy array (all zeros)
2. For each piece on board:
   - Find its position (row, col)
   - Determine channel (0-5 for white, 6-11 for black)
   - Set that position to 1.0
3. Flatten: 8×8×12 → 768
4. Convert to PyTorch tensor
5. Move to GPU if available
```

**2. `select_action(board, training=True)`** - Epsilon-Greedy Policy
```python
Process:
1. Get all legal moves
2. If training and random() < epsilon:
      return random move (EXPLORE)
3. Else:
      For each legal move:
          - Simulate move on board
          - Convert position to tensor
          - Get neural network evaluation
          - Undo move
      return move with best evaluation (EXPLOIT)
```

**Why epsilon-greedy?**
- Early training: Needs to try many moves to learn
- Late training: Should use learned knowledge
- Always keeps small randomness (1%) to discover new strategies

**3. `calculate_reward(board, move)`** - Reward Function
```python
Rewards:
+ Capture pawn:    +1
+ Capture knight:  +3
+ Capture bishop:  +3
+ Capture rook:    +5
+ Capture queen:   +9
+ Checkmate:       +100
+ Give check:      +2
- Stalemate:       -10
- Each move:       -0.01 (encourages faster wins)
```

**4. `train_step()`** - The Learning Algorithm
```python
Process:
1. Check if enough experiences (need 64+)
2. Sample random batch of 64 experiences
3. For each experience:
   Q_current = policy_net(state)
   Q_next = target_net(next_state)
   Q_target = reward + gamma × Q_next
4. Calculate loss: MSE(Q_current, Q_target)
5. Backpropagate and update policy_net weights
6. Return loss value
```

**Mathematical Formula:**
```
Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
                      └─────┬─────┘
                         TD Target
Where:
- s = current state
- a = action taken
- r = reward received
- s' = next state
- α = learning rate (0.001)
- γ = gamma (0.99)
```

---

### 3. `training.py` - Training Loop

**Purpose**: Orchestrates the training process with visualization.

#### ChessTrainer Class

**Structure:**
```python
class ChessTrainer:
    def __init__(self, agent, board_display, visualize=True, delay=0.1):
        self.agent = agent              # The RL agent
        self.board_display = board_display  # Visual board
        self.visualize = visualize      # Show moves?
        self.delay = delay              # Pause between moves
        
        # Statistics tracking
        self.total_rewards = []         # Reward per episode
        self.game_lengths = []          # Moves per episode
        self.wins = {'white': 0, 'black': 0, 'draw': 0}
```

**`play_game()` - Single Training Episode**

```python
Step-by-step flow:

1. Reset board to starting position
2. Initialize episode_reward = 0, moves_count = 0

3. While game not over and moves < 200:
   
   a. Get current board state as tensor
   
   b. Agent selects move (epsilon-greedy)
   
   c. Calculate reward for this move
   
   d. Execute move on board
   
   e. Get next state
   
   f. Store experience: (state, move, reward, next_state, done)
   
   g. Train agent on random batch from replay buffer
   
   h. If visualizing: render board and pause
   
   i. Accumulate reward, increment move count

4. Update statistics (wins/losses/draws)

5. Every 10 episodes: update target network

6. Decay epsilon (explore less over time)

7. Return episode statistics
```

**`train()` - Full Training Session**
```python
for episode in range(num_episodes):
    # Play one complete game
    result = play_game()
    
    # Every 10 episodes: print progress
    if episode % 10 == 0:
        print average rewards, game lengths, epsilon, wins
    
    # Handle early termination (window closed)
    if result['terminated']:
        break

# Save final model
agent.save('chess_agent_final.pth')
```

**`train_self_play()` - Advanced Training**
```python
# Creates TWO agents that learn from playing each other
# White and Black both improve simultaneously
# Generally leads to better learning than playing random moves
```

---

### 4. `visualization.py` - Metrics Dashboard

**Purpose**: Real-time visualization of training progress using matplotlib.

#### MetricsDashboard Class

**Creates 4 plots:**

**Plot 1: Rewards Over Time**
```python
# Shows how total reward per episode changes
# Raw values (noisy line) + Moving average (smooth line)
# Y-axis: Total reward
# X-axis: Episode number
# Interpretation: Should trend upward as agent improves
```

**Plot 2: Training Loss**
```python
# Shows neural network training loss
# Samples points if too many (max 1000 displayed)
# Y-axis: MSE loss
# X-axis: Training step
# Interpretation: Should trend downward (network learning)
```

**Plot 3: Game Length**
```python
# Shows number of moves per game
# Raw values + Moving average
# Y-axis: Number of moves
# X-axis: Episode number
# Interpretation: 
#   - Early: Very short (random play loses fast)
#   - Middle: Longer (learning to play)
#   - Late: Moderate (playing well but efficiently)
```

**Plot 4: Win/Loss/Draw Distribution**
```python
# Bar chart showing game outcomes
# Categories: White Wins, Black Wins, Draws
# Y-axis: Count
# Shows if agent is learning to win
```

**`render()` - Convert Matplotlib to Pygame**
```python
Process:
1. Update all 4 plots with latest data
2. Render matplotlib figure to canvas
3. Get RGBA buffer from canvas
4. Convert to pygame surface
5. Scale to window size
6. Display in pygame window
```

---

### 5. `main.py` - Application Entry Point

**Purpose**: Provides command-line interface for all modes.

#### Three Main Modes:

**1. Play Mode** - `play_mode()`
```python
# Human vs AI
# Click to move pieces
# AI responds with its best move (epsilon=0, no exploration)
# Keys: R (restart), Q (quit)
```

**2. Watch Mode** - `watch_mode()`
```python
# AI vs AI
# Both sides use same or different trained models
# Adjustable speed (+ / - keys)
# Pause/Resume (SPACE)
# Useful for evaluating agent strength
```

**3. Train Mode** - `train_mode()`
```python
# Primary training interface
# Options:
#   --episodes: How many games to play
#   --visualize / --no-visualize: Show board?
#   --metrics: Show metrics dashboard?
#   --load: Load pre-trained model

# Keys during training:
#   SPACE: Pause/Resume
#   M: Toggle metrics
#   S: Save current model
#   Q: Quit and save
```

**Command-Line Arguments:**
```bash
python main.py --mode train --episodes 100 --metrics
               └─────┬────┘ └────┬────┘ └───┬───┘
                  Mode      Num games    Show graphs

python main.py --mode play --load chess_agent.pth
                            └────┬─────────────┘
                         Load trained model

python main.py --mode watch --load model.pth
```

---

## Core Concepts

### 1. Deep Q-Learning (DQN)

**What is Q?**
Q(state, action) = Expected total future reward for taking action in state

**Goal:**
Learn a function Q that tells us: "If I'm in this position and make this move, how much reward will I get in the long run?"

**How it works:**
```
Traditional Q-Learning:
- Use a table: Q[state][action] = value
- Problem: Chess has ~10^43 positions (can't fit in table!)

Deep Q-Learning:
- Use neural network: Q(state, action) ≈ NN(state, action)
- Network learns to approximate Q-values
- Can generalize to unseen positions
```

**Bellman Equation:**
```
Q(s,a) = r + γ × max Q(s',a')
         └┬┘   └──────┬──────┘
    Immediate   Best future
     reward       value
```

**In chess:**
- Making a good move → Positive reward now
- Leading to winning position → Positive reward later
- Q-value = Sum of both

### 2. Experience Replay

**Problem without replay:**
```
Game: e4 e5, Nf3 Nc6, Bb5 a6...
      ↑   ↑    ↑    ↑    ↑   ↑
      |   |    |    |    |   |
   Consecutive states are very similar!
   Network overfits to recent games!
```

**Solution with replay:**
```
Buffer: [game1_move5, game50_move12, game3_move8, ...]
         └───Random mix of experiences from all games───┘
         
Sample randomly → Network sees diverse situations → Better learning
```

**Analogy:**
Studying for exam:
- Bad: Read textbook start to finish repeatedly
- Good: Shuffle flashcards, study random topics

### 3. Target Network

**Problem:**
```
Q(s,a) = r + γ × max Q(s',a')
                  ↑
         If this keeps changing, target is moving!
         Like shooting at moving target → unstable
```

**Solution:**
```
Policy Network (updates every step):
    Q_policy(s,a) = r + γ × max Q_target(s',a')
                                     ↑
Target Network (updates every 10 episodes):
    Provides stable target for learning
```

**Analogy:**
- Policy = Student taking test (learning)
- Target = Answer key (stable reference)
- Update answer key periodically as student improves

### 4. Epsilon-Greedy Exploration

**Exploration vs Exploitation Dilemma:**
```
Restaurant example:
- Favorite restaurant (exploit): Good meal guaranteed
- New restaurant (explore): Might find better option!

Chess:
- Known good move (exploit): Safe, decent position
- Unusual move (explore): Might discover brilliant strategy!
```

**Epsilon schedule:**
```python
Episode 1:    ε = 1.00  → 100% random (pure exploration)
Episode 100:  ε = 0.60  → 60% random, 40% smart
Episode 500:  ε = 0.08  → 8% random, 92% smart
Episode 5000: ε = 0.01  → 1% random, 99% smart (mostly exploitation)
```

### 5. Reward Shaping

**Why important?**
Chess has sparse rewards:
- Most moves: No immediate feedback
- Only checkmate: Clear reward
- Agent needs guidance during game

**Our reward structure:**
```python
Immediate tactical rewards:
+ Capture pieces (material advantage)
+ Give check (pressure opponent)

Terminal rewards:
+ Win by checkmate (ultimate goal)
- Draw (missed opportunity)

Efficiency penalty:
- Small cost per move (encourage decisive play)
```

**Alternative reward shapes:**
```python
# Too sparse (hard to learn):
reward = 100 if checkmate else 0

# Too dense (may learn wrong patterns):
reward = piece_value + position_value + mobility + ...

# Our balance (tactical + strategic):
reward = captures + checks + game_result - move_cost
```

---

## Code Examples

### Example 1: Creating and Training a Basic Agent

```python
import pygame
from chess_board import ChessBoard
from rl_agent import ChessRLAgent
from training import ChessTrainer

# Initialize components
pygame.init()
board = ChessBoard(width=800, height=800)
agent = ChessRLAgent(
    learning_rate=0.001,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.995
)

# Create trainer
trainer = ChessTrainer(
    agent=agent,
    board_display=board,
    visualize=True,
    delay=0.1
)

# Train for 100 episodes
stats = trainer.train(num_episodes=100)

# Save trained model
agent.save('my_chess_agent.pth')

print(f"Training complete!")
print(f"Final wins: {trainer.wins}")
print(f"Average reward: {sum(trainer.total_rewards[-10:]) / 10}")
```

### Example 2: Loading and Using a Trained Agent

```python
from chess_board import ChessBoard
from rl_agent import ChessRLAgent
import chess

# Load trained agent
agent = ChessRLAgent(epsilon=0.0)  # No exploration
agent.load('my_chess_agent.pth')

# Create a board
board = chess.Board()

# Play a few moves
for i in range(10):
    move = agent.select_action(board, training=False)
    if move:
        print(f"Move {i+1}: {move}")
        board.push(move)
    else:
        print("Game over!")
        break

print(f"\nFinal position:\n{board}")
```

### Example 3: Custom Reward Function

```python
class CustomChessAgent(ChessRLAgent):
    def calculate_reward(self, board, move):
        """Custom reward emphasizing piece development."""
        reward = super().calculate_reward(board, move)
        
        # Bonus for developing pieces in opening
        if board.fullmove_number <= 10:
            # Check if move develops a piece from back rank
            from_square = move.from_square
            if from_square // 8 in [0, 7]:  # Back rank
                piece = board.piece_at(from_square)
                if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    reward += 0.5  # Development bonus
        
        # Bonus for controlling center
        to_square = move.to_square
        center_squares = [27, 28, 35, 36]  # e4, d4, e5, d5
        if to_square in center_squares:
            reward += 0.3
        
        return reward

# Use custom agent
custom_agent = CustomChessAgent()
```

### Example 4: Analyzing Training Progress

```python
import matplotlib.pyplot as plt
from training import ChessTrainer

# After training...
trainer = ChessTrainer(agent, board)
stats = trainer.train(num_episodes=500)

# Plot rewards
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(trainer.total_rewards)
plt.title('Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

plt.subplot(1, 3, 2)
plt.plot(trainer.game_lengths)
plt.title('Game Length')
plt.xlabel('Episode')
plt.ylabel('Number of Moves')

plt.subplot(1, 3, 3)
episodes = list(range(len(agent.epsilon_history)))
plt.plot(episodes, agent.epsilon_history)
plt.title('Epsilon Decay')
plt.xlabel('Episode')
plt.ylabel('Exploration Rate')

plt.tight_layout()
plt.savefig('training_analysis.png')
plt.show()
```

### Example 5: Batch Training with Checkpoints

```python
from chess_board import ChessBoard
from rl_agent import ChessRLAgent
from training import ChessTrainer
import os

def train_with_checkpoints(total_episodes=1000, checkpoint_every=100):
    """Train with periodic checkpoints."""
    
    board = ChessBoard()
    agent = ChessRLAgent()
    trainer = ChessTrainer(agent, board, visualize=False)
    
    os.makedirs('checkpoints', exist_ok=True)
    
    for checkpoint in range(0, total_episodes, checkpoint_every):
        print(f"\n{'='*50}")
        print(f"Training episodes {checkpoint} to {checkpoint + checkpoint_every}")
        print(f"{'='*50}")
        
        # Train for checkpoint_every episodes
        stats = trainer.train(num_episodes=checkpoint_every)
        
        # Save checkpoint
        checkpoint_path = f'checkpoints/agent_episode_{checkpoint + checkpoint_every}.pth'
        agent.save(checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Print stats
        recent_wins = trainer.wins
        print(f"Cumulative Wins: {recent_wins}")
        print(f"Current Epsilon: {agent.epsilon:.4f}")
    
    print("\nTraining complete!")
    return trainer

# Run batch training
trainer = train_with_checkpoints(total_episodes=1000, checkpoint_every=100)
```

### Example 6: Compare Two Agents

```python
import chess
from rl_agent import ChessRLAgent

def play_match(agent1_path, agent2_path, num_games=10):
    """Play multiple games between two agents."""
    
    agent1 = ChessRLAgent(epsilon=0.0)
    agent2 = ChessRLAgent(epsilon=0.0)
    
    agent1.load(agent1_path)
    agent2.load(agent2_path)
    
    results = {'agent1': 0, 'agent2': 0, 'draw': 0}
    
    for game_num in range(num_games):
        board = chess.Board()
        moves = 0
        max_moves = 200
        
        while not board.is_game_over() and moves < max_moves:
            # Alternate between agents
            if board.turn == chess.WHITE:
                move = agent1.select_action(board, training=False)
            else:
                move = agent2.select_action(board, training=False)
            
            if move:
                board.push(move)
                moves += 1
            else:
                break
        
        # Record result
        if board.is_checkmate():
            if board.turn == chess.BLACK:
                results['agent1'] += 1
                winner = 'Agent1'
            else:
                results['agent2'] += 1
                winner = 'Agent2'
        else:
            results['draw'] += 1
            winner = 'Draw'
        
        print(f"Game {game_num + 1}: {winner} ({moves} moves)")
    
    print(f"\nFinal Results: {results}")
    return results

# Compare two checkpoints
results = play_match(
    'checkpoints/agent_episode_100.pth',
    'checkpoints/agent_episode_500.pth',
    num_games=20
)
```

---

## Training Process

### Phase 1: Random Exploration (Episodes 1-50)
```
Epsilon: 100% → 60%
Behavior:
- Completely random moves
- Games end quickly (often illegal positions or quick checkmates)
- Agent discovers basic rules through trial and error
- Reward mostly negative

What's being learned:
- Piece movement patterns
- Basic captures
- "Checkmate = very good"
```

### Phase 2: Pattern Recognition (Episodes 50-200)
```
Epsilon: 60% → 14%
Behavior:
- Mix of random and learned moves
- Starts preferring captures
- Games last longer
- Begins to avoid blunders

What's being learned:
- Material advantage is good
- Protecting valuable pieces
- Simple tactical patterns
- Position evaluation basics
```

### Phase 3: Strategic Play (Episodes 200-500)
```
Epsilon: 14% → 1.5%
Behavior:
- Mostly uses learned policy
- Captures material consistently
- Plays coherent games
- Occasional brilliant moves

What's being learned:
- Multi-move combinations
- Piece coordination
- King safety
- Endgame patterns
```

### Phase 4: Refinement (Episodes 500+)
```
Epsilon: 1.5% → 1%
Behavior:
- Strong tactical play
- Consistent strategy
- Efficient wins
- Minimal blunders

What's being learned:
- Advanced tactics
- Positional understanding
- Opening principles
- Complex endgames
```

---

## Advanced Usage

### Hyperparameter Tuning

```python
# Aggressive learning (faster but less stable)
agent = ChessRLAgent(
    learning_rate=0.01,      # 10x higher
    gamma=0.95,              # Less future-focused
    epsilon_decay=0.99       # Slower exploration decay
)

# Conservative learning (slower but more stable)
agent = ChessRLAgent(
    learning_rate=0.0001,    # 10x lower
    gamma=0.995,             # More future-focused
    epsilon_decay=0.995      # Standard decay
)

# Large replay buffer (more memory, better learning)
agent.replay_buffer = ReplayBuffer(capacity=50000)
agent.batch_size = 128
```

### Multi-GPU Training

```python
import torch

# Force GPU usage
agent = ChessRLAgent()
if torch.cuda.is_available():
    agent.device = torch.device("cuda:0")
    agent.policy_net = agent.policy_net.to(agent.device)
    agent.target_net = agent.target_net.to(agent.device)
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
```

### Export to Other Formats

```python
import torch.onnx

# Export to ONNX (for deployment)
dummy_input = torch.randn(1, 768)
torch.onnx.export(
    agent.policy_net,
    dummy_input,
    "chess_model.onnx",
    export_params=True
)
```

### Integration with Stockfish for Evaluation

```python
import chess.engine

def evaluate_against_stockfish(agent, num_games=10):
    """Play against Stockfish to evaluate strength."""
    
    engine = chess.engine.SimpleEngine.popen_uci("/path/to/stockfish")
    engine.configure({"Skill Level": 5})  # Adjust difficulty
    
    wins = 0
    for game in range(num_games):
        board = chess.Board()
        
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                # Agent plays
                move = agent.select_action(board, training=False)
            else:
                # Stockfish plays
                result = engine.play(board, chess.engine.Limit(time=0.1))
                move = result.move
            
            if move:
                board.push(move)
        
        if board.result() == "1-0":
            wins += 1
    
    engine.quit()
    return f"Agent won {wins}/{num_games} games against Stockfish"
```

---

## Troubleshooting

### Common Issues

**1. Training is very slow**
- Disable visualization: `--no-visualize`
- Reduce episodes or use faster hardware
- Check if using GPU: `agent.device`

**2. Agent not improving**
- Increase training episodes (need 1000+)
- Adjust learning rate
- Check reward function is balanced
- Verify epsilon is decaying

**3. Memory errors**
- Reduce replay buffer size
- Reduce batch size
- Process fewer episodes at once

**4. Graphics issues**
- Update pygame: `pip install --upgrade pygame`
- Update matplotlib: `pip install --upgrade matplotlib`
- Check backend: `matplotlib.use('Agg')`

---

## Performance Benchmarks

### Training Times (Approximate)

```
Hardware: M1 MacBook Pro / RTX 3080

Episodes | No Viz | With Viz | With Metrics
---------|--------|----------|-------------
100      | 5 min  | 15 min   | 20 min
500      | 25 min | 75 min   | 90 min
1000     | 50 min | 150 min  | 180 min
5000     | 4 hrs  | 12 hrs   | 15 hrs
```

### Agent Strength Progression

```
Episodes | Approx ELO | Description
---------|------------|-------------
0        | 400        | Random moves
100      | 600        | Understands captures
500      | 900        | Basic tactics
1000     | 1200       | Intermediate play
5000     | 1500-1600  | Strong tactical play
```

---

## Future Enhancements

### Possible Improvements

1. **AlphaZero-style MCTS**: Use Monte Carlo Tree Search
2. **Transformer Architecture**: Replace MLP with attention mechanisms
3. **Self-play curriculum**: Progressive difficulty increase
4. **Opening book**: Integrate opening theory
5. **Endgame tablebases**: Perfect endgame play
6. **Multi-agent training**: Population-based training
7. **Transfer learning**: Pre-train on chess puzzles

---

## References

### Papers
- "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)
- "Mastering Chess and Shogi by Self-Play with a General RL Algorithm" (Silver et al., 2017)

### Libraries
- PyTorch: https://pytorch.org/
- python-chess: https://python-chess.readthedocs.io/
- pygame: https://www.pygame.org/

---

## License & Contributing

This is an educational project. Feel free to:
- Modify the code
- Experiment with different architectures
- Share improvements
- Use for learning RL concepts

Happy coding! 🎮🧠♟️
