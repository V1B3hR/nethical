# Nethical Branding Assets

This directory contains official branding assets for the Nethical project.

## Files

- **nethical_banner.png** (1050x300px): Official banner for headers, social media, and documentation
- **nethical_logo.png** (256x256px): Official logo for icons, avatars, and thumbnails

## Usage

These assets are used throughout the Nethical project:

- **README.md**: Banner at the top, logo under the title
- **Marketplace submissions**: For plugin stores and registries
- **Documentation**: Headers and branding elements
- **External integrations**: Plugin icons and featured images

## Updating Assets

To update the branding assets with new official images:

1. Place the original high-resolution images as:
   - `nethical_banner_original.png` (any size, will be resized to 1050x300)
   - `nethical_logo_original.png` (any size, will be resized to 256x256)

2. Run the resize script:
   ```bash
   python3 scripts/create_branding_assets.py
   ```

3. The script will automatically resize the originals to the correct dimensions

## Guidelines

See [docs/marketplace/README.md](../docs/marketplace/README.md) for:
- Detailed usage guidelines
- Markdown and HTML snippets
- Brand colors
- Marketplace-specific requirements

## License

These branding assets are part of the Nethical project and are licensed under the MIT License.
