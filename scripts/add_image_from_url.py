#!/usr/bin/env python3
"""
Download and process Nethical branding assets from URLs or local files.

Usage:
    python3 add_image_from_url.py --banner <url_or_path> --logo <url_or_path>
"""

import argparse
import sys
from pathlib import Path
from PIL import Image
import requests


def download_image(url_or_path, output_path):
    """
    Download or copy an image from URL or local path.

    Args:
        url_or_path: URL or local file path
        output_path: Where to save the image

    Returns:
        True if successful, False otherwise
    """
    try:
        if url_or_path.startswith(("http://", "https://")):
            # Download from URL
            print(f"Downloading from URL: {url_or_path}")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36"
            }
            response = requests.get(
                url_or_path, headers=headers, timeout=30, allow_redirects=True
            )

            if response.status_code != 200:
                print(f"✗ Failed to download: HTTP {response.status_code}")
                return False

            # Check if it's an image
            content_type = response.headers.get("content-type", "")
            if "image" not in content_type and len(response.content) < 100:
                print("✗ Response doesn't appear to be an image")
                print(f"  Content-Type: {content_type}")
                return False

            # Save the image
            Path(output_path).write_bytes(response.content)
            print(f"✓ Downloaded {len(response.content)} bytes")

        else:
            # Copy from local path
            print(f"Copying from local path: {url_or_path}")
            source = Path(url_or_path)

            if not source.exists():
                print(f"✗ File not found: {url_or_path}")
                return False

            import shutil

            shutil.copy2(source, output_path)
            print("✓ Copied successfully")

        # Verify it's a valid image
        try:
            img = Image.open(output_path)
            print(f"✓ Valid image: {img.size[0]}x{img.size[1]} {img.mode}")
            img.close()
            return True
        except Exception as e:
            print(f"✗ Not a valid image: {e}")
            Path(output_path).unlink(missing_ok=True)
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download and process Nethical branding assets"
    )
    parser.add_argument(
        "--banner",
        help="URL or path to banner image",
        required=False,
    )
    parser.add_argument(
        "--logo",
        help="URL or path to logo image",
        required=False,
    )
    parser.add_argument(
        "--assets-dir",
        default="assets",
        help="Assets directory (default: assets)",
    )

    args = parser.parse_args()

    if not args.banner and not args.logo:
        parser.print_help()
        print("\n✗ Error: Please provide at least one of --banner or --logo")
        sys.exit(1)

    # Get script directory and ensure assets directory exists
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    assets_dir = repo_root / args.assets_dir
    assets_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Nethical Branding Assets Download")
    print("=" * 60)
    print()

    success = True

    # Download banner
    if args.banner:
        print("Processing banner...")
        banner_path = assets_dir / "nethical_banner_original.png"
        if download_image(args.banner, banner_path):
            print(f"✓ Banner saved to: {banner_path}")
        else:
            print("✗ Failed to process banner")
            success = False
        print()

    # Download logo
    if args.logo:
        print("Processing logo...")
        logo_path = assets_dir / "nethical_logo_original.png"
        if download_image(args.logo, logo_path):
            print(f"✓ Logo saved to: {logo_path}")
        else:
            print("✗ Failed to process logo")
            success = False
        print()

    if success:
        print("=" * 60)
        print("✓ All images processed successfully!")
        print("=" * 60)
        print()
        print("Next step: Run the resize script:")
        print(f"  python3 {script_dir / 'create_branding_assets.py'}")
        print()
        return 0
    else:
        print("=" * 60)
        print("✗ Some images failed to process")
        print("=" * 60)
        print()
        print("Please check the errors above and try again.")
        print("Or manually place images as:")
        print(f"  {assets_dir / 'nethical_banner_original.png'}")
        print(f"  {assets_dir / 'nethical_logo_original.png'}")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
