"""Parallel training using multiple processes for RTX 4090."""
import torch
import torch.multiprocessing as mp
import chess
import numpy as np
from rl_agent import ChessRLAgent
from typing import List, Tuple
import time

class ParallelChessTrainer:
    """Multi-process parallel training optimized for RTX 4090."""
    
    def __init__(self, num_workers=8, learning_rate=0.001):
        """
        Args:
            num_workers: Number of parallel game workers (8-16 recommended for RTX 4090)
            learning_rate: Learning rate for the optimizer
        """
        self.num_workers = num_workers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Shared agent (main process)
        self.agent = ChessRLAgent(learning_rate=learning_rate)
        
        # Statistics
        self.total_games = 0
        self.total_rewards = []
        self.game_lengths = []
        self.wins = {'white': 0, 'black': 0, 'draw': 0}
        
        print(f"Parallel Trainer initialized with {num_workers} workers on {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def play_game_worker(self, agent_state: dict, max_moves=200) -> Tuple[List, float, int, str]:
        """Worker function to play one game. Returns experiences, reward, length, outcome."""
        # Create local agent with shared weights
        local_agent = ChessRLAgent()
        local_agent.policy_net.load_state_dict(agent_state['policy_net'])
        local_agent.target_net.load_state_dict(agent_state['target_net'])
        local_agent.epsilon = agent_state['epsilon']
        
        board = chess.Board()
        experiences = []
        total_reward = 0
        moves = 0
        
        while not board.is_game_over() and moves < max_moves:
            # Get state
            state = local_agent.board_to_tensor(board)
            
            # Select action
            move = local_agent.select_action(board, training=True)
            if move is None:
                break
            
            # Calculate reward
            reward = local_agent.calculate_reward(board, move)
            
            # Make move
            board.push(move)
            moves += 1
            
            # Get next state
            next_state = None if board.is_game_over() else local_agent.board_to_tensor(board)
            
            # Store experience
            done = board.is_game_over()
            experiences.append((state, move, reward, next_state, done))
            
            total_reward += reward
        
        # Determine outcome
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                outcome = "white"
            elif result == "0-1":
                outcome = "black"
            else:
                outcome = "draw"
        else:
            outcome = "draw"
        
        return experiences, total_reward, moves, outcome
    
    def train_parallel(self, num_episodes=1000, update_interval=10, save_interval=100):
        """
        Train using parallel game generation.
        
        Args:
            num_episodes: Total number of episodes to train
            update_interval: Games per batch before training
            save_interval: Save model every N episodes
        """
        print(f"\nStarting parallel training for {num_episodes} episodes...")
        print(f"Batch size per update: {update_interval} games")
        print(f"Neural network batch size: {self.agent.batch_size}")
        print(f"Replay buffer capacity: {self.agent.replay_buffer.buffer.maxlen}")
        
        episode = 0
        start_time = time.time()
        
        while episode < num_episodes:
            batch_start = time.time()
            
            # Get current agent state for workers
            agent_state = {
                'policy_net': self.agent.policy_net.state_dict(),
                'target_net': self.agent.target_net.state_dict(),
                'epsilon': self.agent.epsilon
            }
            
            # Play multiple games in parallel (simulated for now - true multiprocessing needs careful CUDA handling)
            # For GPU training, sequential with large batches is often more efficient
            batch_experiences = []
            batch_rewards = []
            batch_lengths = []
            batch_outcomes = []
            
            for _ in range(min(update_interval, num_episodes - episode)):
                exp, reward, length, outcome = self.play_game_worker(agent_state)
                batch_experiences.extend(exp)
                batch_rewards.append(reward)
                batch_lengths.append(length)
                batch_outcomes.append(outcome)
                episode += 1
            
            # Add all experiences to replay buffer
            for state, move, reward, next_state, done in batch_experiences:
                self.agent.replay_buffer.push(state, move, reward, next_state, done)
            
            # Train on multiple batches
            if len(self.agent.replay_buffer) >= self.agent.batch_size:
                num_train_steps = min(len(batch_experiences) // self.agent.batch_size, 50)
                total_loss = 0
                for _ in range(num_train_steps):
                    loss = self.agent.train_step()
                    if loss is not None:
                        total_loss += loss
                        self.agent.losses.append(loss)
                
                avg_loss = total_loss / num_train_steps if num_train_steps > 0 else 0
            else:
                avg_loss = 0
            
            # Update statistics
            self.total_rewards.extend(batch_rewards)
            self.game_lengths.extend(batch_lengths)
            for outcome in batch_outcomes:
                self.wins[outcome] += 1
            
            # Decay epsilon
            self.agent.decay_epsilon()
            
            # Update target network periodically
            if episode % 50 == 0:
                self.agent.update_target_network()
            
            # Print progress
            batch_time = time.time() - batch_start
            games_per_sec = update_interval / batch_time
            
            if episode % update_interval == 0:
                avg_reward = np.mean(batch_rewards)
                avg_length = np.mean(batch_lengths)
                elapsed = time.time() - start_time
                
                print(f"\n{'='*70}")
                print(f"Episode {episode}/{num_episodes} ({episode/num_episodes*100:.1f}%)")
                print(f"  Time: {elapsed:.1f}s | Speed: {games_per_sec:.2f} games/sec")
                print(f"  Avg Reward: {avg_reward:.2f} | Avg Length: {avg_length:.1f} moves")
                print(f"  Loss: {avg_loss:.4f} | Epsilon: {self.agent.epsilon:.3f}")
                print(f"  W/B/D: {self.wins['white']}/{self.wins['black']}/{self.wins['draw']}")
                print(f"  Replay Buffer: {len(self.agent.replay_buffer)}/{self.agent.replay_buffer.buffer.maxlen}")
                
                if torch.cuda.is_available():
                    print(f"  GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB / {torch.cuda.max_memory_allocated()/1e9:.2f} GB peak")
            
            # Save checkpoint
            if episode % save_interval == 0 and episode > 0:
                checkpoint_path = f'chess_agent_ep{episode}.pth'
                self.agent.save(checkpoint_path)
                print(f"\n  [CHECKPOINT] Saved to {checkpoint_path}")
        
        # Final save
        print(f"\n{'='*70}")
        print("Training completed!")
        self.agent.save('chess_agent_final_parallel.pth')
        print("Final model saved to chess_agent_final_parallel.pth")
        
        # Final statistics
        total_time = time.time() - start_time
        print(f"\n=== Final Statistics ===")
        print(f"Total Episodes: {episode}")
        print(f"Total Time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
        print(f"Avg Speed: {episode/total_time:.2f} games/sec")
        print(f"Wins (White): {self.wins['white']} ({self.wins['white']/episode*100:.1f}%)")
        print(f"Wins (Black): {self.wins['black']} ({self.wins['black']/episode*100:.1f}%)")
        print(f"Draws: {self.wins['draw']} ({self.wins['draw']/episode*100:.1f}%)")
        
        if len(self.total_rewards) > 0:
            print(f"Avg Reward: {np.mean(self.total_rewards):.2f}")
            print(f"Avg Game Length: {np.mean(self.game_lengths):.1f} moves")
        
        if torch.cuda.is_available():
            print(f"\nGPU Stats:")
            print(f"  Peak Memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
            print(f"  Total GPU Time: {total_time:.2f}s")


def main():
    """Run parallel training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Parallel Chess RL Training (RTX 4090 Optimized)')
    parser.add_argument('--episodes', type=int, default=10000,
                       help='Number of training episodes (default: 10000)')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of parallel workers (default: 8)')
    parser.add_argument('--batch', type=int, default=10,
                       help='Games per training update (default: 10)')
    parser.add_argument('--save-interval', type=int, default=1000,
                       help='Save checkpoint every N episodes (default: 1000)')
    parser.add_argument('--load', type=str, default=None,
                       help='Load existing model checkpoint')
    parser.add_argument('--lr', type=float, default=0.0005,
                       help='Learning rate (default: 0.0005)')
    
    args = parser.parse_args()
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Training will be slow on CPU.")
        print("Make sure PyTorch is installed with CUDA support:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
    
    # Create trainer
    trainer = ParallelChessTrainer(num_workers=args.workers, learning_rate=args.lr)
    
    # Load existing model if provided
    if args.load:
        try:
            trainer.agent.load(args.load)
            print(f"Loaded model from {args.load}")
        except Exception as e:
            print(f"Could not load model: {e}")
    
    # Train
    trainer.train_parallel(
        num_episodes=args.episodes,
        update_interval=args.batch,
        save_interval=args.save_interval
    )


if __name__ == '__main__':
    main()
