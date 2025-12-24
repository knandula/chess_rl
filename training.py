import pygame
import chess
import time
from typing import Optional
from chess_board import ChessBoard
from rl_agent import ChessRLAgent

class ChessTrainer:
    """Training loop with visualization."""
    
    def __init__(self, agent: ChessRLAgent, board_display: ChessBoard, 
                 visualize=True, delay=0.1):
        self.agent = agent
        self.board_display = board_display
        self.visualize = visualize
        self.delay = delay
        
        # Training stats
        self.episode = 0
        self.total_rewards = []
        self.game_lengths = []
        self.wins = {'white': 0, 'black': 0, 'draw': 0}
    
    def play_game(self, opponent_agent: Optional[ChessRLAgent] = None, 
                  max_moves=200) -> dict:
        """Play one complete game."""
        self.board_display.reset()
        board = self.board_display.board
        
        episode_reward = 0
        moves_count = 0
        game_history = []
        
        while not board.is_game_over() and moves_count < max_moves:
            # Handle pygame events
            if self.visualize:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return {'terminated': True}
            
            # Current player
            if board.turn == chess.WHITE:
                current_agent = self.agent
            else:
                current_agent = opponent_agent if opponent_agent else self.agent
            
            # Get state
            state = current_agent.board_to_tensor(board)
            
            # Select action
            move = current_agent.select_action(board, training=True)
            
            if move is None:
                break
            
            # Calculate reward
            reward = current_agent.calculate_reward(board, move)
            
            # Make move
            board.push(move)
            moves_count += 1
            
            # Get next state
            next_state = None if board.is_game_over() else current_agent.board_to_tensor(board)
            
            # Store experience
            done = board.is_game_over()
            current_agent.replay_buffer.push(state, move, reward, next_state, done)
            
            # Train
            loss = current_agent.train_step()
            if loss is not None:
                current_agent.losses.append(loss)
            
            episode_reward += reward
            
            # Visualize
            if self.visualize:
                self.board_display.render()
                time.sleep(self.delay)
            
            game_history.append({
                'move': move,
                'reward': reward,
                'state': state
            })
        
        # Game over
        result = self._process_game_result(board)
        
        # Update target network periodically
        if self.episode % 10 == 0:
            self.agent.update_target_network()
            if opponent_agent:
                opponent_agent.update_target_network()
        
        # Decay epsilon
        self.agent.decay_epsilon()
        if opponent_agent:
            opponent_agent.decay_epsilon()
        
        # Store statistics
        self.total_rewards.append(episode_reward)
        self.game_lengths.append(moves_count)
        
        return {
            'terminated': False,
            'reward': episode_reward,
            'moves': moves_count,
            'result': result,
            'game_history': game_history
        }
    
    def _process_game_result(self, board: chess.Board) -> str:
        """Process and record game result."""
        if board.is_checkmate():
            winner = 'black' if board.turn == chess.WHITE else 'white'
            self.wins[winner] += 1
            return winner
        else:
            self.wins['draw'] += 1
            return 'draw'
    
    def train(self, num_episodes=100, update_interval=10):
        """Train the agent for multiple episodes."""
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Device: {self.agent.device}")
        
        for episode in range(num_episodes):
            self.episode = episode
            
            # Play game
            result = self.play_game()
            
            if result.get('terminated'):
                break
            
            # Print progress
            if episode % update_interval == 0:
                avg_reward = sum(self.total_rewards[-update_interval:]) / min(update_interval, len(self.total_rewards))
                avg_length = sum(self.game_lengths[-update_interval:]) / min(update_interval, len(self.game_lengths))
                
                print(f"\nEpisode {episode}/{num_episodes}")
                print(f"  Avg Reward: {avg_reward:.2f}")
                print(f"  Avg Game Length: {avg_length:.1f}")
                print(f"  Epsilon: {self.agent.epsilon:.3f}")
                print(f"  Wins - White: {self.wins['white']}, Black: {self.wins['black']}, Draw: {self.wins['draw']}")
                if self.agent.losses:
                    print(f"  Avg Loss: {sum(self.agent.losses[-100:]) / min(100, len(self.agent.losses)):.4f}")
        
        print("\nTraining completed!")
        return {
            'total_rewards': self.total_rewards,
            'game_lengths': self.game_lengths,
            'wins': self.wins,
            'losses': self.agent.losses
        }
    
    def train_self_play(self, num_episodes=100, update_interval=10):
        """Train agent through self-play."""
        print(f"Starting self-play training for {num_episodes} episodes...")
        
        # Create opponent agent with same architecture
        opponent = ChessRLAgent(
            learning_rate=0.001,
            gamma=0.99,
            epsilon=self.agent.epsilon
        )
        
        for episode in range(num_episodes):
            self.episode = episode
            
            # Play game with both agents learning
            result = self.play_game(opponent_agent=opponent)
            
            if result.get('terminated'):
                break
            
            # Print progress
            if episode % update_interval == 0:
                avg_reward = sum(self.total_rewards[-update_interval:]) / min(update_interval, len(self.total_rewards))
                avg_length = sum(self.game_lengths[-update_interval:]) / min(update_interval, len(self.game_lengths))
                
                print(f"\nEpisode {episode}/{num_episodes}")
                print(f"  Avg Reward: {avg_reward:.2f}")
                print(f"  Avg Game Length: {avg_length:.1f}")
                print(f"  Epsilon: {self.agent.epsilon:.3f}")
                print(f"  Wins - White: {self.wins['white']}, Black: {self.wins['black']}, Draw: {self.wins['draw']}")
                if self.agent.losses:
                    print(f"  Avg Loss: {sum(self.agent.losses[-100:]) / min(100, len(self.agent.losses)):.4f}")
        
        print("\nSelf-play training completed!")
        return {
            'total_rewards': self.total_rewards,
            'game_lengths': self.game_lengths,
            'wins': self.wins,
            'losses': self.agent.losses
        }
