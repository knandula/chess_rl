import pygame
import chess
import os
from typing import Optional, Tuple

class ChessBoard:
    """Visual chess board using pygame."""
    
    def __init__(self, width=800, height=800):
        pygame.init()
        self.width = width
        self.height = height
        self.square_size = width // 8
        
        # Create display
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Chess RL - Learning Visualized")
        
        # Colors
        self.LIGHT_SQUARE = (240, 217, 181)
        self.DARK_SQUARE = (181, 136, 99)
        self.HIGHLIGHT_COLOR = (186, 202, 68, 128)
        self.SELECTED_COLOR = (246, 246, 105, 200)
        
        # Chess board
        self.board = chess.Board()
        
        # Selection
        self.selected_square = None
        self.valid_moves = []
        
        # Last move tracking for arrow visualization
        self.last_move = None
        
        # Load piece images
        self.load_pieces()
        
    def load_pieces(self):
        """Load chess piece images from PNG files."""
        self.pieces = {}
        pieces_dir = "pieces"
        
        # Map piece symbols to filenames
        piece_files = {
            'P': 'wP', 'N': 'wN', 'B': 'wB', 'R': 'wR', 'Q': 'wQ', 'K': 'wK',
            'p': 'bP', 'n': 'bN', 'b': 'bB', 'r': 'bR', 'q': 'bQ', 'k': 'bK'
        }
        
        piece_size = int(self.square_size * 0.8)
        
        for symbol, filename in piece_files.items():
            png_path = os.path.join(pieces_dir, f"{filename}.png")
            
            if os.path.exists(png_path):
                try:
                    # Load PNG and scale it
                    image = pygame.image.load(png_path)
                    image = pygame.transform.smoothscale(image, (piece_size, piece_size))
                    self.pieces[symbol] = image
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        if not self.pieces:
            print("Warning: No chess pieces loaded! Run download_pieces.py and convert_pieces.py first.")
    
    def draw_board(self):
        """Draw the chess board."""
        for row in range(8):
            for col in range(8):
                # Determine square color
                is_light = (row + col) % 2 == 0
                color = self.LIGHT_SQUARE if is_light else self.DARK_SQUARE
                
                # Draw square
                rect = pygame.Rect(
                    col * self.square_size,
                    row * self.square_size,
                    self.square_size,
                    self.square_size
                )
                pygame.draw.rect(self.screen, color, rect)
                
                # Draw coordinates
                if col == 0:
                    rank = str(8 - row)
                    coord_font = pygame.font.Font(None, 20)
                    text = coord_font.render(rank, True, 
                                            self.DARK_SQUARE if is_light else self.LIGHT_SQUARE)
                    self.screen.blit(text, (col * self.square_size + 5, 
                                          row * self.square_size + 5))
                if row == 7:
                    file = chr(ord('a') + col)
                    coord_font = pygame.font.Font(None, 20)
                    text = coord_font.render(file, True,
                                            self.DARK_SQUARE if is_light else self.LIGHT_SQUARE)
                    self.screen.blit(text, (col * self.square_size + self.square_size - 15,
                                          row * self.square_size + self.square_size - 20))
    
    def highlight_square(self, square: int, color: Tuple[int, int, int, int]):
        """Highlight a square with transparency."""
        row = 7 - (square // 8)
        col = square % 8
        
        s = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
        s.fill(color)
        self.screen.blit(s, (col * self.square_size, row * self.square_size))
    
    def draw_pieces(self):
        """Draw chess pieces on the board using images."""
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row = 7 - (square // 8)
                col = square % 8
                
                # Get piece image
                symbol = piece.symbol()
                piece_image = self.pieces.get(symbol)
                
                if piece_image:
                    # Calculate position (center the image)
                    x = col * self.square_size + (self.square_size - piece_image.get_width()) // 2
                    y = row * self.square_size + (self.square_size - piece_image.get_height()) // 2
                    
                    # Draw the piece
                    self.screen.blit(piece_image, (x, y))
    
    def draw_arrow(self, from_square: int, to_square: int, color=(50, 150, 50), width=8):
        """Draw an arrow from one square to another."""
        # Calculate start and end positions
        from_row = 7 - (from_square // 8)
        from_col = from_square % 8
        to_row = 7 - (to_square // 8)
        to_col = to_square % 8
        
        start_x = from_col * self.square_size + self.square_size // 2
        start_y = from_row * self.square_size + self.square_size // 2
        end_x = to_col * self.square_size + self.square_size // 2
        end_y = to_row * self.square_size + self.square_size // 2
        
        # Draw arrow line
        pygame.draw.line(self.screen, color, (start_x, start_y), (end_x, end_y), width)
        
        # Draw arrowhead
        import math
        angle = math.atan2(end_y - start_y, end_x - start_x)
        arrow_length = self.square_size // 3
        arrow_angle = math.pi / 6  # 30 degrees
        
        # Left side of arrowhead
        left_x = end_x - arrow_length * math.cos(angle - arrow_angle)
        left_y = end_y - arrow_length * math.sin(angle - arrow_angle)
        
        # Right side of arrowhead
        right_x = end_x - arrow_length * math.cos(angle + arrow_angle)
        right_y = end_y - arrow_length * math.sin(angle + arrow_angle)
        
        # Draw filled triangle for arrowhead
        pygame.draw.polygon(self.screen, color, [
            (end_x, end_y),
            (left_x, left_y),
            (right_x, right_y)
        ])
    
    def draw_highlights(self):
        """Draw highlights for selected square and valid moves."""
        if self.selected_square is not None:
            self.highlight_square(self.selected_square, self.SELECTED_COLOR)
        
        for move in self.valid_moves:
            self.highlight_square(move.to_square, self.HIGHLIGHT_COLOR)
    
    def render(self):
        """Render the complete board."""
        self.draw_board()
        self.draw_highlights()
        self.draw_pieces()
        
        # Draw arrow for last move
        if self.last_move is not None:
            self.draw_arrow(self.last_move.from_square, self.last_move.to_square)
        
        pygame.display.flip()
    
    def get_square_from_mouse(self, pos: Tuple[int, int]) -> Optional[int]:
        """Convert mouse position to chess square."""
        x, y = pos
        col = x // self.square_size
        row = 7 - (y // self.square_size)
        
        if 0 <= col < 8 and 0 <= row < 8:
            return chess.square(col, row)
        return None
    
    def handle_click(self, square: int) -> Optional[chess.Move]:
        """Handle square click and return move if valid."""
        if self.selected_square is None:
            # Select a piece
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.selected_square = square
                self.valid_moves = [
                    move for move in self.board.legal_moves
                    if move.from_square == square
                ]
        else:
            # Try to make a move
            move = None
            for valid_move in self.valid_moves:
                if valid_move.to_square == square:
                    move = valid_move
                    break
            
            # Reset selection
            self.selected_square = None
            self.valid_moves = []
            
            return move
        
        return None
    
    def make_move(self, move: chess.Move) -> bool:
        """Make a move on the board."""
        if move in self.board.legal_moves:
            self.last_move = move  # Store for arrow visualization
            self.board.push(move)
            self.selected_square = None
            self.valid_moves = []
            return True
        return False
    
    def reset(self):
        """Reset the board to starting position."""
        self.board.reset()
        self.selected_square = None
        self.valid_moves = []
        self.last_move = None
    
    def is_game_over(self) -> bool:
        """Check if game is over."""
        return self.board.is_game_over()
    
    def get_result(self) -> str:
        """Get game result."""
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            return f"{winner} wins by checkmate!"
        elif self.board.is_stalemate():
            return "Draw by stalemate"
        elif self.board.is_insufficient_material():
            return "Draw by insufficient material"
        elif self.board.is_fifty_moves():
            return "Draw by fifty-move rule"
        elif self.board.is_repetition():
            return "Draw by repetition"
        return "Game over"
