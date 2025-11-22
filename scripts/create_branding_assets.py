#!/usr/bin/env python3
"""
Script to create or resize Nethical branding assets.

This script creates placeholder branding assets for the Nethical project.
IMPORTANT: Replace these placeholders with actual branded images!

Required dimensions:
- Banner: 1050x300px (nethical_banner.png)
- Logo: 256x256px (nethical_logo.png)
"""

from PIL import Image, ImageDraw, ImageFont
import os
import sys


def find_font(font_name, size, fallback_default=True):
    """
    Find a font across different operating systems.

    Args:
        font_name: Name/style of font (e.g., 'bold', 'regular')
        size: Font size
        fallback_default: Whether to fall back to default font if not found

    Returns:
        ImageFont object
    """
    # Common font paths across different operating systems
    font_paths = []

    if sys.platform == "win32":
        # Windows font locations
        font_paths = [
            "C:\\Windows\\Fonts\\arial.ttf",
            "C:\\Windows\\Fonts\\arialbd.ttf",
            "C:\\Windows\\Fonts\\segoeui.ttf",
            "C:\\Windows\\Fonts\\segoeuib.ttf",
        ]
    elif sys.platform == "darwin":
        # macOS font locations
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNSText.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
    else:
        # Linux font locations
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]

    # Try to load each font in order
    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, size)
        except (OSError, IOError):
            continue

    # Fall back to default font if requested
    if fallback_default:
        return ImageFont.load_default()

    raise RuntimeError(f"Could not find suitable font for {font_name}")


def create_banner_placeholder(output_path, width=1050, height=300):
    """Create a placeholder banner image."""
    # Create image with dark background
    img = Image.new("RGB", (width, height), color="#1a1a2e")
    draw = ImageDraw.Draw(img)

    # Add a simple visual element
    # Draw gradient-like rectangles
    colors = ["#16213e", "#0f3460", "#533483"]
    for i, color in enumerate(colors):
        x_start = i * (width // 3)
        x_end = (i + 1) * (width // 3)
        draw.rectangle([x_start, 0, x_end, height], fill=color)

    # Add text
    text = "NETHICAL"
    subtext = "AI Safety & Ethics Governance"

    # Load fonts using cross-platform helper
    font = find_font("bold", 60)
    subfont = find_font("regular", 24)

    # Calculate text position (centered)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2 - 30

    # Draw text with shadow effect
    draw.text((x + 2, y + 2), text, fill="#000000", font=font)
    draw.text((x, y), text, fill="#e94560", font=font)

    # Draw subtext
    bbox2 = draw.textbbox((0, 0), subtext, font=subfont)
    subtext_width = bbox2[2] - bbox2[0]
    x_sub = (width - subtext_width) // 2
    y_sub = y + text_height + 20

    draw.text((x_sub + 1, y_sub + 1), subtext, fill="#000000", font=subfont)
    draw.text((x_sub, y_sub), subtext, fill="#ffffff", font=subfont)

    # Save image
    img.save(output_path, "PNG")
    print(f"Created placeholder banner: {output_path} ({width}x{height})")


def create_logo_placeholder(output_path, size=256):
    """Create a placeholder logo image."""
    # Create image with transparent background
    img = Image.new("RGBA", (size, size), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw a shield-like shape
    margin = size // 8

    # Background circle
    draw.ellipse([margin, margin, size - margin, size - margin], fill="#0f3460")

    # Inner circle for contrast
    inner_margin = size // 4
    draw.ellipse(
        [inner_margin, inner_margin, size - inner_margin, size - inner_margin],
        fill="#533483",
    )

    # Draw "N" letter
    font = find_font("bold", size // 3)

    text = "N"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size - text_width) // 2
    y = (size - text_height) // 2

    # Draw text
    draw.text((x, y), text, fill="#e94560", font=font)

    # Save image
    img.save(output_path, "PNG")
    print(f"Created placeholder logo: {output_path} ({size}x{size})")


def resize_image(input_path, output_path, target_size):
    """Resize an image to target dimensions."""
    try:
        img = Image.open(input_path)

        # Resize with high-quality resampling
        if isinstance(target_size, tuple):
            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        else:
            img_resized = img.resize(
                (target_size, target_size), Image.Resampling.LANCZOS
            )

        # Save with optimization
        img_resized.save(output_path, "PNG", optimize=True)
        print(f"Resized {input_path} to {target_size} -> {output_path}")
        return True
    except Exception as e:
        print(f"Error resizing {input_path}: {e}")
        return False


def main():
    """Main function to create branding assets."""
    # Get repository root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    assets_dir = os.path.join(repo_root, "assets")

    # Ensure assets directory exists
    os.makedirs(assets_dir, exist_ok=True)

    # Define paths
    banner_path = os.path.join(assets_dir, "nethical_banner.png")
    logo_path = os.path.join(assets_dir, "nethical_logo.png")

    # Check if original images exist
    # (they should be placed in assets/ with _original suffix)
    original_banner = os.path.join(assets_dir, "nethical_banner_original.png")
    original_logo = os.path.join(assets_dir, "nethical_logo_original.png")

    # If originals exist, resize them
    if os.path.exists(original_banner):
        print("Found original banner, resizing...")
        resize_image(original_banner, banner_path, (1050, 300))
    else:
        print("No original banner found, creating placeholder...")
        create_banner_placeholder(banner_path, 1050, 300)
        print("\nIMPORTANT: This is a PLACEHOLDER!")
        print(f"Replace with branded image: {original_banner}")

    if os.path.exists(original_logo):
        print("Found original logo, resizing...")
        resize_image(original_logo, logo_path, (256, 256))
    else:
        print("No original logo found, creating placeholder...")
        create_logo_placeholder(logo_path, 256)
        print("\nIMPORTANT: This is a PLACEHOLDER!")
        print(f"Replace with branded image: {original_logo}")

    print("\n" + "=" * 60)
    print("Branding assets created successfully!")
    print(f"Banner: {banner_path}")
    print(f"Logo: {logo_path}")
    print("=" * 60)

    if not (os.path.exists(original_banner) and os.path.exists(original_logo)):
        print("\nNOTE: Placeholder images were created.")
        print("To use actual branded images:")
        print(f"  1. Place original banner as: {original_banner}")
        print(f"  2. Place original logo as: {original_logo}")
        print("  3. Run this script again to resize them properly")


if __name__ == "__main__":
    main()
