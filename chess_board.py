import pygame
import chess
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
        
        # Load piece images
        self.load_pieces()
        
    def load_pieces(self):
        """Load chess piece images."""
        # Pieces will be drawn as shapes, no need for font
        self.pieces = {
            'p': 'pawn', 'r': 'rook', 'n': 'knight', 
            'b': 'bishop', 'q': 'queen', 'k': 'king',
            'P': 'pawn', 'R': 'rook', 'N': 'knight',
            'B': 'bishop', 'Q': 'queen', 'K': 'king'
        }
    
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
        """Draw chess pieces on the board."""
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row = 7 - (square // 8)
                col = square % 8
                
                # Get piece type
                symbol = piece.symbol()
                piece_type = self.pieces[symbol]
                
                # Calculate position
                center_x = col * self.square_size + self.square_size // 2
                center_y = row * self.square_size + self.square_size // 2
                size = int(self.square_size * 0.35)
                
                # Colors
                if piece.color == chess.WHITE:
                    fill_color = (240, 240, 240)
                    outline_color = (60, 60, 60)
                else:
                    fill_color = (40, 40, 40)
                    outline_color = (220, 220, 220)
                
                # Draw different shapes based on piece type
                if piece_type == 'pawn':
                    # Small circle
                    pygame.draw.circle(self.screen, fill_color, (center_x, center_y), size)
                    pygame.draw.circle(self.screen, outline_color, (center_x, center_y), size, 3)
                
                elif piece_type == 'rook':
                    # Rectangle with battlements
                    rect = pygame.Rect(center_x - size, center_y - size, size * 2, size * 2)
                    pygame.draw.rect(self.screen, fill_color, rect)
                    pygame.draw.rect(self.screen, outline_color, rect, 3)
                    # Battlements
                    batt_size = size // 3
                    pygame.draw.rect(self.screen, outline_color, 
                                   (center_x - size, center_y - size, batt_size, batt_size), 3)
                    pygame.draw.rect(self.screen, outline_color,
                                   (center_x + size - batt_size, center_y - size, batt_size, batt_size), 3)
                
                elif piece_type == 'knight':
                    # Triangle
                    points = [
                        (center_x, center_y - size),
                        (center_x - size, center_y + size),
                        (center_x + size, center_y + size)
                    ]
                    pygame.draw.polygon(self.screen, fill_color, points)
                    pygame.draw.polygon(self.screen, outline_color, points, 3)
                
                elif piece_type == 'bishop':
                    # Diamond
                    points = [
                        (center_x, center_y - size),
                        (center_x + size, center_y),
                        (center_x, center_y + size),
                        (center_x - size, center_y)
                    ]
                    pygame.draw.polygon(self.screen, fill_color, points)
                    pygame.draw.polygon(self.screen, outline_color, points, 3)
                
                elif piece_type == 'queen':
                    # Circle with crown points
                    pygame.draw.circle(self.screen, fill_color, (center_x, center_y), size)
                    pygame.draw.circle(self.screen, outline_color, (center_x, center_y), size, 3)
                    # Crown points (5 small circles on top)
                    crown_radius = size // 5
                    for i in range(5):
                        angle = (i * 72 - 90) * 3.14159 / 180
                        px = center_x + int(size * 0.7 * pygame.math.Vector2(1, 0).rotate_rad(angle).x)
                        py = center_y + int(size * 0.7 * pygame.math.Vector2(1, 0).rotate_rad(angle).y)
                        pygame.draw.circle(self.screen, outline_color, (px, py), crown_radius)
                
                elif piece_type == 'king':
                    # Circle with cross on top
                    pygame.draw.circle(self.screen, fill_color, (center_x, center_y), size)
                    pygame.draw.circle(self.screen, outline_color, (center_x, center_y), size, 3)
                    # Cross
                    cross_size = size // 2
                    pygame.draw.line(self.screen, outline_color,
                                   (center_x, center_y - cross_size),
                                   (center_x, center_y + cross_size), 4)
                    pygame.draw.line(self.screen, outline_color,
                                   (center_x - cross_size, center_y),
                                   (center_x + cross_size, center_y), 4)
    
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
