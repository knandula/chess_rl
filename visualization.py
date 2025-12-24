import matplotlib
matplotlib.use('Agg')  # Use Agg backend for compatibility with pygame on macOS
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import pygame
from typing import List

class MetricsDashboard:
    """Real-time visualization of training metrics."""
    
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        
        # Create pygame window for metrics
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Training Metrics Dashboard")
        
        # Create matplotlib figure
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8))
        self.fig.tight_layout(pad=3.0)
        
        # Colors
        self.bg_color = (30, 30, 30)
        self.text_color = (255, 255, 255)
        
    def update_plots(self, trainer_stats: dict, current_episode: int):
        """Update all plots with current statistics."""
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot 1: Rewards over time
        if trainer_stats.get('total_rewards'):
            rewards = trainer_stats['total_rewards']
            episodes = list(range(len(rewards)))
            
            self.axes[0, 0].plot(episodes, rewards, 'b-', alpha=0.3, label='Raw')
            
            # Moving average
            if len(rewards) > 10:
                window = min(20, len(rewards))
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                self.axes[0, 0].plot(range(window-1, len(rewards)), moving_avg, 'r-', 
                                    linewidth=2, label=f'{window}-game avg')
            
            self.axes[0, 0].set_xlabel('Episode')
            self.axes[0, 0].set_ylabel('Total Reward')
            self.axes[0, 0].set_title('Rewards Over Time')
            self.axes[0, 0].legend()
            self.axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Loss over time
        if trainer_stats.get('losses'):
            losses = trainer_stats['losses']
            steps = list(range(len(losses)))
            
            # Sample if too many points
            if len(losses) > 1000:
                sample_idx = np.linspace(0, len(losses)-1, 1000, dtype=int)
                losses = [losses[i] for i in sample_idx]
                steps = list(sample_idx)
            
            self.axes[0, 1].plot(steps, losses, 'g-', alpha=0.4)
            
            # Moving average
            if len(losses) > 10:
                window = min(50, len(losses))
                moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
                self.axes[0, 1].plot(range(window-1, len(losses)), moving_avg, 'orange', 
                                    linewidth=2, label=f'{window}-step avg')
            
            self.axes[0, 1].set_xlabel('Training Step')
            self.axes[0, 1].set_ylabel('Loss')
            self.axes[0, 1].set_title('Training Loss')
            self.axes[0, 1].legend()
            self.axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Game length over time
        if trainer_stats.get('game_lengths'):
            lengths = trainer_stats['game_lengths']
            episodes = list(range(len(lengths)))
            
            self.axes[1, 0].plot(episodes, lengths, 'purple', alpha=0.3, label='Raw')
            
            # Moving average
            if len(lengths) > 10:
                window = min(20, len(lengths))
                moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
                self.axes[1, 0].plot(range(window-1, len(lengths)), moving_avg, 'cyan', 
                                    linewidth=2, label=f'{window}-game avg')
            
            self.axes[1, 0].set_xlabel('Episode')
            self.axes[1, 0].set_ylabel('Moves')
            self.axes[1, 0].set_title('Game Length')
            self.axes[1, 0].legend()
            self.axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Win/Loss/Draw distribution
        if trainer_stats.get('wins'):
            wins_data = trainer_stats['wins']
            categories = ['White Wins', 'Black Wins', 'Draws']
            values = [wins_data['white'], wins_data['black'], wins_data['draw']]
            colors = ['lightblue', 'lightcoral', 'lightgray']
            
            bars = self.axes[1, 1].bar(categories, values, color=colors, edgecolor='black')
            self.axes[1, 1].set_ylabel('Count')
            self.axes[1, 1].set_title('Game Outcomes')
            self.axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                self.axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                                    f'{int(height)}',
                                    ha='center', va='bottom')
        
        # Add overall title with episode info
        self.fig.suptitle(f'Training Progress - Episode {current_episode}', 
                         fontsize=14, fontweight='bold')
    
    def render(self, trainer_stats: dict, current_episode: int):
        """Render the dashboard to pygame window."""
        # Update plots
        self.update_plots(trainer_stats, current_episode)
        
        # Convert matplotlib figure to pygame surface
        canvas = FigureCanvasAgg(self.fig)
        canvas.draw()
        
        # Get the RGBA buffer and convert to RGB
        buf = canvas.buffer_rgba()
        size = canvas.get_width_height()
        surf = pygame.image.frombuffer(buf, size, "RGBA")
        
        # Scale to fit window
        surf = pygame.transform.scale(surf, (self.width, self.height))
        
        # Draw to screen
        self.screen.fill(self.bg_color)
        self.screen.blit(surf, (0, 0))
        
        pygame.display.flip()
    
    def close(self):
        """Close the dashboard."""
        plt.close(self.fig)


class LiveTrainingVisualizer:
    """Combines chess board and metrics dashboard."""
    
    def __init__(self, board_width=800, board_height=800, 
                 metrics_width=800, metrics_height=600):
        # Initialize pygame
        pygame.init()
        
        # Create two windows (this is a simplified version)
        # For true multi-window, you'd need threading or separate processes
        self.board_width = board_width
        self.board_height = board_height
        self.metrics_width = metrics_width
        self.metrics_height = metrics_height
        
        self.current_view = 'board'  # 'board' or 'metrics'
        
    def toggle_view(self):
        """Toggle between board and metrics view."""
        if self.current_view == 'board':
            self.current_view = 'metrics'
        else:
            self.current_view = 'board'
    
    def show_summary(self, screen, trainer_stats: dict, episode: int):
        """Show training summary on screen."""
        font = pygame.font.Font(None, 36)
        small_font = pygame.font.Font(None, 24)
        
        # Background
        screen.fill((30, 30, 30))
        
        y_offset = 50
        
        # Title
        title = font.render(f'Episode {episode} Summary', True, (255, 255, 255))
        screen.blit(title, (50, y_offset))
        y_offset += 60
        
        # Stats
        if trainer_stats.get('total_rewards'):
            recent_rewards = trainer_stats['total_rewards'][-10:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            text = small_font.render(f'Avg Reward (last 10): {avg_reward:.2f}', 
                                    True, (255, 255, 255))
            screen.blit(text, (50, y_offset))
            y_offset += 40
        
        if trainer_stats.get('game_lengths'):
            recent_lengths = trainer_stats['game_lengths'][-10:]
            avg_length = sum(recent_lengths) / len(recent_lengths)
            text = small_font.render(f'Avg Game Length (last 10): {avg_length:.1f}', 
                                    True, (255, 255, 255))
            screen.blit(text, (50, y_offset))
            y_offset += 40
        
        if trainer_stats.get('wins'):
            wins = trainer_stats['wins']
            text = small_font.render(
                f"Results - White: {wins['white']}, Black: {wins['black']}, Draw: {wins['draw']}", 
                True, (255, 255, 255))
            screen.blit(text, (50, y_offset))
            y_offset += 40
        
        # Instructions
        y_offset += 40
        inst = small_font.render('Press SPACE to continue, M for metrics, Q to quit', 
                                True, (200, 200, 200))
        screen.blit(inst, (50, y_offset))
        
        pygame.display.flip()
