import pygame
import chess

class PieceRenderer:
    """Render chess pieces using pygame drawing primitives."""
    
    @staticmethod
    def draw_pawn(surface, center_x, center_y, size, is_white):
        """Draw a pawn piece."""
        color = (255, 255, 255) if is_white else (0, 0, 0)
        outline = (0, 0, 0) if is_white else (255, 255, 255)
        
        # Base (circle at bottom)
        base_y = center_y + size // 3
        pygame.draw.circle(surface, color, (center_x, base_y), size // 3)
        pygame.draw.circle(surface, outline, (center_x, base_y), size // 3, 2)
        
        # Head (smaller circle on top)
        head_y = center_y - size // 4
        pygame.draw.circle(surface, color, (center_x, head_y), size // 4)
        pygame.draw.circle(surface, outline, (center_x, head_y), size // 4, 2)
    
    @staticmethod
    def draw_rook(surface, center_x, center_y, size, is_white):
        """Draw a rook piece."""
        color = (255, 255, 255) if is_white else (0, 0, 0)
        outline = (0, 0, 0) if is_white else (255, 255, 255)
        
        # Main body (rectangle)
        width = size // 2
        height = int(size * 0.8)
        rect = pygame.Rect(center_x - width // 2, center_y - height // 2, width, height)
        pygame.draw.rect(surface, color, rect)
        pygame.draw.rect(surface, outline, rect, 2)
        
        # Battlements (top)
        battlement_size = width // 3
        top_y = center_y - height // 2
        for i in [-1, 1]:
            batt_x = center_x + i * width // 4
            batt_rect = pygame.Rect(batt_x - battlement_size // 2, top_y - battlement_size, 
                                   battlement_size, battlement_size)
            pygame.draw.rect(surface, color, batt_rect)
            pygame.draw.rect(surface, outline, batt_rect, 2)
    
    @staticmethod
    def draw_knight(surface, center_x, center_y, size, is_white):
        """Draw a knight piece."""
        color = (255, 255, 255) if is_white else (0, 0, 0)
        outline = (0, 0, 0) if is_white else (255, 255, 255)
        
        # Knight shape (polygon resembling horse head)
        points = [
            (center_x - size // 3, center_y + size // 2),  # Bottom left
            (center_x - size // 4, center_y - size // 3),  # Left ear
            (center_x, center_y - size // 2),              # Top
            (center_x + size // 3, center_y),              # Snout
            (center_x + size // 4, center_y + size // 2)   # Bottom right
        ]
        pygame.draw.polygon(surface, color, points)
        pygame.draw.polygon(surface, outline, points, 2)
        
        # Eye
        eye_x = center_x + size // 6
        eye_y = center_y - size // 6
        pygame.draw.circle(surface, outline, (eye_x, eye_y), 3)
    
    @staticmethod
    def draw_bishop(surface, center_x, center_y, size, is_white):
        """Draw a bishop piece."""
        color = (255, 255, 255) if is_white else (0, 0, 0)
        outline = (0, 0, 0) if is_white else (255, 255, 255)
        
        # Body (elongated diamond)
        points = [
            (center_x, center_y - int(size * 0.6)),        # Top
            (center_x + size // 3, center_y),              # Right
            (center_x, center_y + int(size * 0.5)),        # Bottom
            (center_x - size // 3, center_y)               # Left
        ]
        pygame.draw.polygon(surface, color, points)
        pygame.draw.polygon(surface, outline, points, 2)
        
        # Slit at top
        slit_y = center_y - int(size * 0.5)
        pygame.draw.line(surface, outline, 
                        (center_x - 5, slit_y), (center_x + 5, slit_y), 3)
    
    @staticmethod
    def draw_queen(surface, center_x, center_y, size, is_white):
        """Draw a queen piece."""
        color = (255, 255, 255) if is_white else (0, 0, 0)
        outline = (0, 0, 0) if is_white else (255, 255, 255)
        
        # Base circle
        pygame.draw.circle(surface, color, (center_x, center_y + size // 4), size // 3)
        pygame.draw.circle(surface, outline, (center_x, center_y + size // 4), size // 3, 2)
        
        # Crown points (5 circles)
        crown_y = center_y - size // 3
        crown_radius = size // 6
        positions = [
            (center_x, crown_y - size // 6),                    # Top
            (center_x - size // 3, crown_y),                    # Left
            (center_x + size // 3, crown_y),                    # Right
            (center_x - size // 5, crown_y + size // 8),        # Bottom left
            (center_x + size // 5, crown_y + size // 8)         # Bottom right
        ]
        
        for pos_x, pos_y in positions:
            pygame.draw.circle(surface, color, (pos_x, pos_y), crown_radius)
            pygame.draw.circle(surface, outline, (pos_x, pos_y), crown_radius, 2)
    
    @staticmethod
    def draw_king(surface, center_x, center_y, size, is_white):
        """Draw a king piece."""
        color = (255, 255, 255) if is_white else (0, 0, 0)
        outline = (0, 0, 0) if is_white else (255, 255, 255)
        
        # Base circle
        pygame.draw.circle(surface, color, (center_x, center_y + size // 6), int(size * 0.35))
        pygame.draw.circle(surface, outline, (center_x, center_y + size // 6), int(size * 0.35), 2)
        
        # Crown
        crown_size = size // 2
        crown_y = center_y - size // 3
        
        # Vertical line of cross
        pygame.draw.line(surface, outline,
                        (center_x, crown_y - crown_size // 2),
                        (center_x, crown_y + crown_size // 2), 4)
        
        # Horizontal line of cross
        pygame.draw.line(surface, outline,
                        (center_x - crown_size // 2, crown_y),
                        (center_x + crown_size // 2, crown_y), 4)
        
        # Cross endpoints (small circles)
        for dx, dy in [(0, -crown_size//2), (0, crown_size//2), 
                       (-crown_size//2, 0), (crown_size//2, 0)]:
            pygame.draw.circle(surface, color, 
                             (center_x + dx, crown_y + dy), 4)
            pygame.draw.circle(surface, outline, 
                             (center_x + dx, crown_y + dy), 4, 2)
    
    @classmethod
    def draw_piece(cls, surface, piece_type, center_x, center_y, size, is_white):
        """Draw a chess piece based on type."""
        draw_funcs = {
            'pawn': cls.draw_pawn,
            'rook': cls.draw_rook,
            'knight': cls.draw_knight,
            'bishop': cls.draw_bishop,
            'queen': cls.draw_queen,
            'king': cls.draw_king
        }
        
        func = draw_funcs.get(piece_type)
        if func:
            func(surface, center_x, center_y, size, is_white)
