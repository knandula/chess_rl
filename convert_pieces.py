"""Convert SVG chess pieces to PNG format."""
import os
from PIL import Image
import subprocess

def svg_to_png_convert():
    """Convert SVG files to PNG using system tools."""
    pieces_dir = "pieces"
    
    svg_files = [f for f in os.listdir(pieces_dir) if f.endswith('.svg')]
    
    print("Converting SVG to PNG...")
    for svg_file in svg_files:
        svg_path = os.path.join(pieces_dir, svg_file)
        png_file = svg_file.replace('.svg', '.png')
        png_path = os.path.join(pieces_dir, png_file)
        
        if os.path.exists(png_path):
            print(f"  {png_file} already exists, skipping...")
            continue
        
        try:
            # Use rsvg-convert or qlmanage (macOS) or imagemagick
            print(f"  Converting {svg_file} to PNG...")
            
            # Try qlmanage (built-in on macOS)
            cmd = [
                'qlmanage', '-t', '-s', '100', '-o', pieces_dir, svg_path
            ]
            result = subprocess.run(cmd, capture_output=True)
            
            # Rename the output
            temp_png = svg_path + '.png'
            if os.path.exists(temp_png):
                os.rename(temp_png, png_path)
                print(f"    ✓ Created {png_file}")
            else:
                print(f"    ✗ Failed to convert {svg_file}")
                
        except Exception as e:
            print(f"  Error converting {svg_file}: {e}")
    
    print("Conversion complete!")

if __name__ == "__main__":
    svg_to_png_convert()
