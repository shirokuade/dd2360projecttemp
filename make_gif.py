#!/usr/bin/env python3
"""
Generate a GIF from render progress frames.
Works on Google Colab without ImageMagick.

Usage:
    python make_gif.py [output_name] [fps]

Example:
    python make_gif.py render_progress 1
"""

import os
import sys
import glob

def ppm_to_array(filename):
    """Read a PPM file and return as numpy array."""
    import numpy as np

    with open(filename, 'rb') as f:
        # Read header
        header = f.readline().decode().strip()
        if header != 'P3' and header != 'P6':
            # Handle P3 format (ASCII)
            f.seek(0)
            lines = f.read().decode().split('\n')

            # Skip comments and get dimensions
            idx = 0
            while lines[idx].startswith('#') or lines[idx].strip() == 'P3':
                idx += 1

            dims = lines[idx].split()
            width, height = int(dims[0]), int(dims[1])
            idx += 1
            max_val = int(lines[idx])
            idx += 1

            # Read pixel data
            pixels = []
            for line in lines[idx:]:
                pixels.extend(line.split())

            pixels = [int(p) for p in pixels if p]
            img = np.array(pixels, dtype=np.uint8).reshape((height, width, 3))
            return img

    return None

def main():
    output_name = sys.argv[1] if len(sys.argv) > 1 else "render_progress"
    fps = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    # Find all frame files
    frame_files = sorted(glob.glob("frames/frame_*.ppm"))

    if not frame_files:
        print("No frames found in frames/ directory")
        return

    print(f"Found {len(frame_files)} frames")

    try:
        from PIL import Image
        import numpy as np

        images = []
        for i, f in enumerate(frame_files):
            print(f"\rLoading frame {i+1}/{len(frame_files)}", end="", flush=True)
            img_array = ppm_to_array(f)
            if img_array is not None:
                images.append(Image.fromarray(img_array))

        print(f"\nSaving GIF as {output_name}.gif...")

        # Calculate duration per frame (in milliseconds)
        duration = int(1000 / fps)

        # Save as GIF
        images[0].save(
            f"{output_name}.gif",
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )

        print(f"GIF saved: {output_name}.gif")
        print(f"  - {len(images)} frames")
        print(f"  - {fps} fps ({duration}ms per frame)")
        print(f"  - Total duration: {len(images) / fps:.1f} seconds")

    except ImportError:
        print("PIL not found. Install with: pip install Pillow")
        print("\nAlternatively, use ImageMagick:")
        print(f"  convert -delay {100//fps} -loop 0 frames/frame_*.ppm {output_name}.gif")

if __name__ == "__main__":
    main()
