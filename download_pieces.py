import os
import urllib.request

def download_chess_pieces():
    """Download chess piece images from GitHub raw content."""
    
    # Create pieces directory
    pieces_dir = "pieces"
    if not os.path.exists(pieces_dir):
        os.makedirs(pieces_dir)
    
    # Base URL - using a reliable public CDN for chess pieces
    base_url = "https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/cburnett/"
    
    # Piece filenames
    pieces = {
        'wP': 'wP.svg',
        'wN': 'wN.svg',
        'wB': 'wB.svg',
        'wR': 'wR.svg',
        'wQ': 'wQ.svg',
        'wK': 'wK.svg',
        'bP': 'bP.svg',
        'bN': 'bN.svg',
        'bB': 'bB.svg',
        'bR': 'bR.svg',
        'bQ': 'bQ.svg',
        'bK': 'bK.svg'
    }
    
    print("Downloading chess piece images from Lichess (open source)...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    for piece_name, filename in pieces.items():
        file_path = os.path.join(pieces_dir, f"{piece_name}.svg")
        
        if os.path.exists(file_path):
            print(f"  {piece_name}.svg already exists, skipping...")
            continue
        
        try:
            full_url = base_url + filename
            print(f"  Downloading {piece_name}.svg...")
            req = urllib.request.Request(full_url, headers=headers)
            with urllib.request.urlopen(req) as response:
                with open(file_path, 'wb') as f:
                    f.write(response.read())
        except Exception as e:
            print(f"  Error downloading {piece_name}: {e}")
            return False
    
    print("All chess pieces downloaded successfully!")
    print("\nNote: SVG files downloaded. You'll need to install pygame-ce or cairosvg to use them.")
    print("Or we can convert them to PNG format.")
    return True

if __name__ == "__main__":
    download_chess_pieces()
