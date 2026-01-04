# Marketplace Branding Assets

Official branding assets for Nethical marketplace and plugin submissions.

## Overview

This directory contains official branding resources for developers submitting Nethical plugins to marketplaces, registries, and documentation sites.

## Available Assets

### Banner
![Nethical Banner](../../assets/nethical_banner.png)

- **File**: `assets/nethical_banner.png`
- **Dimensions**: 1050x300 pixels
- **Format**: PNG
- **Use Cases**: 
  - Repository headers
  - Marketplace listings
  - Documentation headers
  - Social media posts
  - Blog articles

### Logo
<p align="center">
  <img src="../../assets/nethical_logo.png" alt="Nethical Logo" width="128" height="128">
</p>

- **File**: `assets/nethical_logo.png`
- **Dimensions**: 256x256 pixels
- **Format**: PNG (with transparency)
- **Use Cases**:
  - Plugin icons
  - Marketplace thumbnails
  - App icons
  - Favicons (can be resized)
  - Avatar images

## Usage Guidelines

### For Plugin Developers

When submitting a plugin that integrates with or extends Nethical:

1. **Include Attribution**: Use the Nethical logo in your plugin documentation
2. **Maintain Brand Consistency**: Don't modify colors or proportions
3. **Clear Association**: Make it clear your plugin integrates with Nethical
4. **Marketplace Listings**: Use provided assets in marketplace submissions

### Markdown Usage

Copy these snippets for use in your documentation:

#### Banner (full width)
```markdown
![Nethical Banner](https://raw.githubusercontent.com/V1B3hR/nethical/main/assets/nethical_banner.png)
```

#### Logo (centered, medium size)
```markdown
<p align="center">
  <img src="https://raw.githubusercontent.com/V1B3hR/nethical/main/assets/nethical_logo.png" alt="Nethical Logo" width="128" height="128">
</p>
```

#### Logo (inline)
```markdown
![Nethical Logo](https://raw.githubusercontent.com/V1B3hR/nethical/main/assets/nethical_logo.png)
```

### HTML Usage

For HTML documentation or web pages:

#### Banner
```html
<img src="https://raw.githubusercontent.com/V1B3hR/nethical/main/assets/nethical_banner.png" 
     alt="Nethical - AI Safety & Ethics Governance" 
     style="max-width: 100%; height: auto;">
```

#### Logo
```html
<img src="https://raw.githubusercontent.com/V1B3hR/nethical/main/assets/nethical_logo.png" 
     alt="Nethical Logo" 
     width="128" 
     height="128">
```

## Marketplace-Specific Requirements

### OpenAI Plugin Store

When submitting to the OpenAI plugin store, use:
- **Plugin Icon**: Logo (256x256)
- **Featured Image**: Banner (1050x300)

Include in your `ai-plugin.json`:
```json
{
  "logo_url": "https://raw.githubusercontent.com/V1B3hR/nethical/main/assets/nethical_logo.png",
  "name_for_human": "Nethical AI Safety Guard"
}
```

### LangChain Hub

For LangChain Hub submissions:
- **Tool Icon**: Logo (256x256)
- **Repository Banner**: Banner (1050x300)

### HuggingFace Spaces

For HuggingFace Spaces:
- **App Thumbnail**: Logo (256x256)
- **README Banner**: Banner (1050x300)

### Anthropic Claude

For Claude integration documentation:
- **Tool Icon**: Logo (256x256)
- **Documentation Header**: Banner (1050x300)

## Brand Colors

For complementary design elements:

- **Primary**: `#e94560` (Coral Red)
- **Secondary**: `#0f3460` (Deep Blue)
- **Accent**: `#533483` (Purple)
- **Background Dark**: `#1a1a2e` (Almost Black)
- **Background Light**: `#16213e` (Navy Blue)

## File Locations

All branding assets are located in the repository at:

```
nethical/
├── assets/
│   ├── nethical_banner.png       # 1050x300 banner
│   └── nethical_logo.png          # 256x256 logo
└── docs/
    └── marketplace/
        └── README.md              # This file
```

## License and Attribution

The Nethical branding assets are part of the Nethical project, licensed under the MIT License.

When using these assets:
- ✅ **DO**: Use for Nethical-related projects, integrations, and documentation
- ✅ **DO**: Attribute to the Nethical project
- ✅ **DO**: Link back to the Nethical repository
- ❌ **DON'T**: Modify or alter the assets
- ❌ **DON'T**: Use for unrelated projects
- ❌ **DON'T**: Imply official endorsement without permission

## Need Different Sizes?

If you need the assets in different dimensions for specific marketplace requirements:

1. Use the provided Python script to resize:
   ```bash
   cd scripts/
   python3 create_branding_assets.py
   ```

2. Or use standard image tools:
   ```bash
   # ImageMagick example
   convert assets/nethical_logo.png -resize 512x512 logo_512.png
   
   # For favicons
   convert assets/nethical_logo.png -resize 32x32 favicon.ico
   ```

## Support

For questions about branding asset usage:
- Open an issue in the [Nethical repository](https://github.com/V1B3hR/nethical/issues)
- Check the [CONTRIBUTING.md](../../CONTRIBUTING.md) guide
- Review the [Marketplace Registration Guide](../MARKETPLACE_REGISTRATION_GUIDE.md)

## Updates

**Last Updated**: November 2025

The branding assets may be updated periodically. Check this repository for the latest versions.

---

**Note**: These are placeholder images for demonstration. Replace with official branded assets when available by placing originals as `assets/nethical_banner_original.png` and `assets/nethical_logo_original.png`, then running the resize script.
