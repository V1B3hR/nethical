# Marketplace Branding Guidelines

This guide provides official branding assets and guidelines for submitting Nethical to plugin registries, trust mark programs, and marketplace platforms.

## Official Branding Assets

Nethical provides official branding assets to ensure consistent representation across all platforms:

### Logo

**Location:** `assets/nethical_logo.png`

- **Format:** PNG with transparency support
- **Dimensions:** 512x512 pixels (square)
- **Usage:** Profile pictures, app icons, plugin listings, and marketplace thumbnails
- **Design:** Shield with lock symbol in blue and gold, representing security and trust

### Banner

**Location:** `assets/nethical_banner.png`

- **Format:** PNG
- **Dimensions:** 1200x300 pixels (4:1 ratio)
- **Usage:** Repository headers, marketplace cover images, documentation headers
- **Design:** Full branding with logo, name, and tagline

## Branding Usage Guidelines

### When to Use the Logo

✅ **Recommended Uses:**
- Plugin registry profile pictures
- Marketplace app icons (resize to platform requirements)
- Documentation and tutorial headers
- Social media profiles representing Nethical
- Trust mark and certification submissions
- Integration showcase galleries

❌ **Avoid:**
- Modifying colors or proportions
- Adding effects or filters that obscure the design
- Using low-resolution versions
- Combining with competing product logos

### When to Use the Banner

✅ **Recommended Uses:**
- GitHub repository header (already in README)
- Marketplace cover images and hero sections
- Blog post headers about Nethical
- Conference presentation slides
- Marketing materials and landing pages

❌ **Avoid:**
- Cropping or stretching the banner
- Overlaying text that obscures the branding
- Using on backgrounds that reduce contrast

## Marketplace Submission Checklist

When submitting Nethical to a marketplace or plugin registry, use this checklist:

### Required Assets

- [ ] **Logo**: Use `assets/nethical_logo.png` (resize if needed, maintaining aspect ratio)
- [ ] **Banner/Cover**: Use `assets/nethical_banner.png` (crop/resize as needed for platform)
- [ ] **Name**: "Nethical"
- [ ] **Tagline**: "Safety, Ethics and More for AI Agents"
- [ ] **Category**: Security / AI Governance / Safety Tools
- [ ] **Description**: See [Marketplace Registration Guide](../MARKETPLACE_REGISTRATION_GUIDE.md)

### Image Specifications by Platform

Different platforms have varying image requirements. Here are recommended approaches:

#### OpenAI Plugin Store
- **Logo**: 512x512 PNG (use as-is)
- **Banner**: Not required, but can be used in documentation

#### Anthropic Claude Tools
- **Logo**: 256x256 PNG (resize from 512x512)
- **Icon**: Use logo cropped to shield symbol only if needed

#### xAI Grok Marketplace
- **Logo**: 512x512 PNG (use as-is)
- **Banner**: 1200x300 PNG (use as-is)

#### Google Gemini
- **Logo**: 512x512 PNG (use as-is)
- **Feature Graphic**: Use banner or create custom 1024x500 version

#### LangChain Hub
- **Logo**: 200x200 PNG (resize from 512x512)
- **No banner required**

#### HuggingFace Spaces
- **Logo**: 512x512 PNG (use as-is)
- **Banner**: 1200x300 PNG (use as-is)

### Resizing Guidelines

If a platform requires different dimensions:

1. **For Logos**: 
   - Maintain square aspect ratio (1:1)
   - Use high-quality scaling (Lanczos or bicubic)
   - Preserve transparency if supported
   - Acceptable sizes: 64x64, 128x128, 256x256, 512x512, 1024x1024

2. **For Banners**:
   - Maintain 4:1 aspect ratio when possible
   - Center-crop if different ratio is required
   - Ensure text remains readable after resizing
   - Acceptable sizes: 600x150, 1200x300, 1920x480

### Quick Resize Commands

Using ImageMagick (if available):

```bash
# Resize logo to 256x256
convert assets/nethical_logo.png -resize 256x256 nethical_logo_256.png

# Resize banner to 1024x256
convert assets/nethical_banner.png -resize 1024x256 nethical_banner_1024.png
```

Using Python (Pillow):

```python
from PIL import Image

# Resize logo
logo = Image.open('assets/nethical_logo.png')
logo_resized = logo.resize((256, 256), Image.LANCZOS)
logo_resized.save('nethical_logo_256.png')

# Resize banner
banner = Image.open('assets/nethical_banner.png')
banner_resized = banner.resize((1024, 256), Image.LANCZOS)
banner_resized.save('nethical_banner_1024.png')
```

## Brand Colors

For custom graphics or platform customization:

- **Primary Blue**: `#3b82f6` (Shield/Primary elements)
- **Dark Blue**: `#0f172a` (Background)
- **Gold**: `#fbbf24` (Lock/Accent)
- **Light Blue**: `#60a5fa` (Text highlights)
- **Gray**: `#94a3b8` (Subtitle/Secondary text)

## Trust Mark & Certification Usage

When submitting for trust marks or certifications:

1. **Use Official Logo**: Always use `assets/nethical_logo.png`
2. **Reference Repository**: Link to https://github.com/V1B3hR/nethical
3. **Official Documentation**: Reference docs in this repository
4. **Compliance Documents**: See `docs/compliance/` for GDPR, NIST, OWASP references

## Platform-Specific Guidance

### AI Plugin Directories

For AI assistant plugin directories (ChatGPT, Claude, etc.):

- Use **logo** as the plugin icon
- Reference **manifest files** in the repository root:
  - `ai-plugin.json` (OpenAI/Anthropic)
  - `grok-manifest.json` (xAI)
  - `gemini-manifest.json` (Google)
- Include link to `docs/MARKETPLACE_REGISTRATION_GUIDE.md`

### Security Tool Registries

For security and cybersecurity tool catalogs:

- Emphasize **safety governance** and **ethical AI** capabilities
- Use both logo and banner for visual impact
- Reference security documentation in `docs/security/`
- Link to `SECURITY.md` for vulnerability reporting

### Developer Marketplaces

For developer tool marketplaces (npm, PyPI adjacent tools):

- Use **logo** for package icon
- Use **banner** in package documentation
- Reference installation instructions in README
- Link to `docs/PLUGIN_DEVELOPER_GUIDE.md` for extensions

## Contact & Support

For branding questions or custom asset requests:

- **Repository**: https://github.com/V1B3hR/nethical
- **Documentation**: See `docs/` directory
- **Compliance**: See `docs/compliance/` directory

## Version History

- **v1.0** (2025-11-22): Initial branding assets and guidelines
  - Created official logo (512x512)
  - Created official banner (1200x300)
  - Established usage guidelines
  - Platform-specific recommendations

---

**Note**: These are the official branding assets for Nethical. Using consistent branding across all platforms helps build trust and recognition in the AI safety and security community.
