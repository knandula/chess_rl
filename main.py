import pygame
import argparse
import sys
import os
import chess
from chess_board import ChessBoard
from rl_agent import ChessRLAgent
from training import ChessTrainer
from visualization import MetricsDashboard, LiveTrainingVisualizer

def play_mode(agent_path=None):
    """Play against the AI."""
    print("Starting Play Mode...")
    
    # Initialize board
    board = ChessBoard()
    
    # Initialize agent
    agent = ChessRLAgent(epsilon=0.0)  # No exploration in play mode
    
    # Auto-load parallel-trained model if available
    if agent_path is None and os.path.exists('chess_agent_final_parallel.pth'):
        agent_path = 'chess_agent_final_parallel.pth'
        print("Found parallel-trained model, loading automatically...")
    
    if agent_path:
        try:
            agent.load(agent_path)
            print(f"Loaded agent from {agent_path}")
        except:
            print("Could not load agent, using random agent")
    
    # Game loop
    clock = pygame.time.Clock()
    running = True
    player_color = chess.WHITE
    
    print("\nPlay Mode Controls:")
    print("- Click to select and move pieces")
    print("- You are playing as White")
    print("- Press R to restart")
    print("- Press Q to quit")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    board.reset()
                    print("\nGame reset!")
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if board.board.turn == player_color and not board.is_game_over():
                    pos = pygame.mouse.get_pos()
                    square = board.get_square_from_mouse(pos)
                    
                    if square is not None:
                        move = board.handle_click(square)
                        if move:
                            board.make_move(move)
        
        # AI move
        if board.board.turn != player_color and not board.is_game_over():
            move = agent.select_action(board.board, training=False)
            if move:
                board.make_move(move)
        
        # Check game over
        if board.is_game_over():
            result = board.get_result()
            print(f"\n{result}")
            print("Press R to restart or Q to quit")
        
        # Render
        board.render()
        clock.tick(60)
    
    pygame.quit()

def watch_mode(agent_path=None):
    """Watch AI play against itself."""
    print("Starting Watch Mode...")
    
    # Initialize board
    board = ChessBoard()
    
    # Initialize agents
    agent_white = ChessRLAgent(epsilon=0.1)
    agent_black = ChessRLAgent(epsilon=0.1)
    
    # Auto-load parallel-trained model if available
    if agent_path is None and os.path.exists('chess_agent_final_parallel.pth'):
        agent_path = 'chess_agent_final_parallel.pth'
        print("Found parallel-trained model, loading automatically...")
    
    if agent_path:
        try:
            agent_white.load(agent_path)
            agent_black.load(agent_path)
            print(f"Loaded agents from {agent_path}")
        except:
            print("Could not load agents, using random agents")
    
    # Game loop
    clock = pygame.time.Clock()
    running = True
    game_speed = 1.0
    paused = False
    
    print("\nWatch Mode Controls:")
    print("- Press SPACE to pause/resume")
    print("- Press R to restart")
    print("- Press + to speed up")
    print("- Press - to slow down")
    print("- Press Q to quit")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    board.reset()
                    print("\nGame reset!")
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    game_speed = min(5.0, game_speed + 0.5)
                    print(f"Speed: {game_speed}x")
                elif event.key == pygame.K_MINUS:
                    game_speed = max(0.5, game_speed - 0.5)
                    print(f"Speed: {game_speed}x")
        
        # AI moves
        if not paused and not board.is_game_over():
            current_agent = agent_white if board.board.turn == chess.WHITE else agent_black
            move = current_agent.select_action(board.board, training=False)
            if move:
                board.make_move(move)
                pygame.time.wait(int(500 / game_speed))
        
        # Check game over
        if board.is_game_over():
            result = board.get_result()
            print(f"\n{result}")
            print("Press R to restart or Q to quit")
        
        # Render
        board.render()
        clock.tick(60)
    
    pygame.quit()

def train_mode(episodes=100, visualize=True, show_metrics=False, load_path=None):
    """Train the AI with visualization."""
    print("Starting Training Mode...")
    
    # Initialize components
    board = ChessBoard()
    agent = ChessRLAgent()
    
    # Load existing model if provided
    if load_path:
        try:
            agent.load(load_path)
            print(f"Loaded existing model from {load_path}")
            print("Continuing training from saved checkpoint...")
        except Exception as e:
            print(f"Could not load model from {load_path}: {e}")
            print("Starting training from scratch...")
    
    trainer = ChessTrainer(agent, board, visualize=visualize, delay=0.01)
    
    # Optional metrics dashboard
    metrics_dashboard = None
    if show_metrics:
        metrics_dashboard = MetricsDashboard()
    
    # Training loop
    clock = pygame.time.Clock()
    running = True
    episode = 0
    
    print("\nTraining Mode Controls:")
    print("- Press SPACE to pause/resume")
    print("- Press M to toggle metrics dashboard")
    print("- Press S to save model")
    print("- Press Q to quit")
    print(f"\nTraining for {episodes} episodes...")
    
    paused = False
    show_metrics_flag = show_metrics
    
    while running and episode < episodes:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print("Training paused" if paused else "Training resumed")
                elif event.key == pygame.K_m:
                    show_metrics_flag = not show_metrics_flag
                    if show_metrics_flag and metrics_dashboard is None:
                        metrics_dashboard = MetricsDashboard()
                    print("Metrics" + (" shown" if show_metrics_flag else " hidden"))
                elif event.key == pygame.K_s:
                    agent.save('chess_agent.pth')
                    print("Model saved to chess_agent.pth")
        
        if not paused:
            # Play one game
            result = trainer.play_game()
            
            if result.get('terminated'):
                running = False
                break
            
            episode += 1
            
            # Update metrics
            if show_metrics_flag and metrics_dashboard and episode % 5 == 0:
                stats = {
                    'total_rewards': trainer.total_rewards,
                    'game_lengths': trainer.game_lengths,
                    'wins': trainer.wins,
                    'losses': agent.losses
                }
                metrics_dashboard.render(stats, episode)
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = sum(trainer.total_rewards[-10:]) / min(10, len(trainer.total_rewards))
                avg_length = sum(trainer.game_lengths[-10:]) / min(10, len(trainer.game_lengths))
                
                print(f"\nEpisode {episode}/{episodes}")
                print(f"  Avg Reward: {avg_reward:.2f}")
                print(f"  Avg Length: {avg_length:.1f}")
                print(f"  Epsilon: {agent.epsilon:.3f}")
                print(f"  W/B/D: {trainer.wins['white']}/{trainer.wins['black']}/{trainer.wins['draw']}")
        
        clock.tick(60)
    
    # Save final model
    print("\nTraining completed!")
    agent.save('chess_agent_final.pth')
    print("Final model saved to chess_agent_final.pth")
    
    # Show final statistics
    print("\n=== Final Statistics ===")
    print(f"Total Episodes: {episode}")
    print(f"Total Wins (White): {trainer.wins['white']}")
    print(f"Total Wins (Black): {trainer.wins['black']}")
    print(f"Total Draws: {trainer.wins['draw']}")
    
    if metrics_dashboard:
        print("\nClose metrics window to exit...")
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
            pygame.time.wait(100)
    
    pygame.quit()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Chess Reinforcement Learning')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'play', 'watch'],
                       help='Mode to run: train, play, or watch')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of training episodes')
    parser.add_argument('--load', type=str, default=None,
                       help='Path to load agent model from')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Disable visualization during training')
    parser.add_argument('--metrics', action='store_true',
                       help='Show metrics dashboard during training')
    
    
    args = parser.parse_args()
    load_path=args.load

    print("=" * 60)
    print("Chess Reinforcement Learning - Visualized Training")
    print("=" * 60)
    
    if args.mode == 'train':
        train_mode(
            episodes=args.episodes,
            visualize=not args.no_visualize,
            show_metrics=args.metrics
        )
    elif args.mode == 'play':
        play_mode(agent_path=args.load)
    elif args.mode == 'watch':
        watch_mode(agent_path=args.load)

if __name__ == '__main__':
    main()
