import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import chess
import random
from collections import deque
from typing import List, Tuple, Optional

class ChessNet(nn.Module):
    """Neural network for evaluating chess positions."""
    
    def __init__(self, input_size=768, hidden_size=2048):
        super(ChessNet, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for training."""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class ChessRLAgent:
    """Reinforcement Learning agent for chess."""
    
    def __init__(self, learning_rate=0.001, gamma=0.99, epsilon=1.0, 
                 epsilon_min=0.01, epsilon_decay=0.995):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy_net = ChessNet().to(self.device)
        self.target_net = ChessNet().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Training parameters
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Replay buffer (scaled for RTX 4090)
        self.replay_buffer = ReplayBuffer(capacity=100000)
        self.batch_size = 512
        
        # Statistics
        self.losses = []
        self.rewards_history = []
        self.epsilon_history = []
    
    def board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        """Convert chess board to neural network input."""
        # Create a 8x8x12 representation (12 piece types, 8x8 board)
        state = np.zeros((8, 8, 12), dtype=np.float32)
        
        piece_idx = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row = square // 8
                col = square % 8
                idx = piece_idx[piece.piece_type]
                if piece.color == chess.BLACK:
                    idx += 6
                state[row, col, idx] = 1.0
        
        # Flatten to vector
        state = state.flatten()
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)
    
    def get_legal_moves_list(self, board: chess.Board) -> List[chess.Move]:
        """Get list of legal moves."""
        return list(board.legal_moves)
    
    def select_action(self, board: chess.Board, training=True) -> Optional[chess.Move]:
        """Select action using epsilon-greedy policy (GPU-optimized batch evaluation)."""
        legal_moves = self.get_legal_moves_list(board)
        
        if not legal_moves:
            return None
        
        # Epsilon-greedy
        if training and random.random() < self.epsilon:
            return random.choice(legal_moves)
        
        # Evaluate all legal moves in batch (GPU-optimized)
        with torch.no_grad():
            # Create batch of all positions after legal moves
            states_batch = []
            for move in legal_moves:
                board.push(move)
                states_batch.append(self.board_to_tensor(board))
                board.pop()
            
            # Evaluate all positions in one forward pass
            if states_batch:
                states_tensor = torch.cat(states_batch, dim=0)
                values = self.policy_net(states_tensor).squeeze()
                
                # Negate values (opponent's perspective)
                values = -values
                
                # Get best move
                best_idx = torch.argmax(values).item()
                best_move = legal_moves[best_idx]
            else:
                best_move = legal_moves[0]
        
        return best_move
    
    def calculate_reward(self, board: chess.Board, move: chess.Move) -> float:
        """Calculate reward for a move."""
        reward = 0.0
        
        # Material values
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        
        # Capture reward
        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                reward += piece_values[captured_piece.piece_type]
        
        # Make move to check game state
        board.push(move)
        
        # Game ending rewards
        if board.is_checkmate():
            reward += 100  # Win
        elif board.is_stalemate() or board.is_insufficient_material():
            reward -= 10  # Draw (slightly negative)
        elif board.is_check():
            reward += 2  # Check bonus
        
        # Small penalty for each move (encourage faster wins)
        reward -= 0.01
        
        # Pop move
        board.pop()
        
        return reward
    
    def train_step(self):
        """Perform one training step."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.cat(states)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states).squeeze()
        
        # Next Q values (batched for GPU)
        with torch.no_grad():
            # Filter non-None next states
            valid_next_states = [ns for ns in next_states if ns is not None]
            valid_indices = [i for i, ns in enumerate(next_states) if ns is not None]
            
            next_q_values = torch.zeros(self.batch_size).to(self.device)
            if valid_next_states:
                # Batch process all valid next states
                next_states_tensor = torch.cat(valid_next_states, dim=0)
                next_values = self.target_net(next_states_tensor).squeeze()
                
                # Assign back to correct positions
                for idx, value in zip(valid_indices, next_values):
                    next_q_values[idx] = value
        
        # Target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)
    
    def save(self, path):
        """Save model."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        """Load model."""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
