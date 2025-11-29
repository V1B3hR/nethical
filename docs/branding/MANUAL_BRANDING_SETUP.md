# Manual Branding Assets Setup

## Issue with Automatic Download

The branding assets cannot be automatically downloaded from GitHub user-attachments URLs due to access restrictions. Please follow the manual setup process below.

## Manual Setup Instructions

### Step 1: Obtain the Original Images

You have two options:

#### Option A: Download from GitHub Issue/PR
1. Navigate to the GitHub issue or PR where the images were uploaded
2. Right-click on each image
3. Select "Save image as..." or "Copy image"
4. Save the files to your computer

#### Option B: Use Provided URLs
If you have direct URLs to the images, you can download them:

**Banner Image URL (provided):**
```
https://github.com/user-attachments/assets/2b6f01c6-bca5-4882-ab24-59c12c699486
```

**Logo Image URL:**
(Please provide the logo URL)

### Step 2: Place Images in Assets Directory

Save the downloaded images to the repository:

```bash
# Navigate to the repository
cd /home/runner/work/nethical/nethical

# Copy or move the images to the assets directory with these exact names:
cp /path/to/downloaded/banner.png assets/nethical_banner_original.png
cp /path/to/downloaded/logo.png assets/nethical_logo_original.png
```

### Step 3: Resize the Images

Run the automated resize script:

```bash
python3 scripts/create_branding_assets.py
```

This will:
- Detect the original images
- Resize the banner to 1050x300px
- Resize the logo to 256x256px
- Save them as `assets/nethical_banner.png` and `assets/nethical_logo.png`
- Preserve high quality with proper resampling

### Step 4: Verify the Results

Check that the images have been created with correct dimensions:

```bash
# View the images
ls -lh assets/nethical_*.png

# Verify dimensions (requires Python with Pillow)
python3 -c "from PIL import Image; b=Image.open('assets/nethical_banner.png'); l=Image.open('assets/nethical_logo.png'); print(f'Banner: {b.size}'); print(f'Logo: {l.size}')"
```

Expected output:
```
Banner: (1050, 300)
Logo: (256, 256)
```

### Step 5: Commit the Changes

```bash
# Add the new images
git add assets/nethical_banner.png assets/nethical_logo.png

# Optionally, add the originals too (if you want to keep them in the repo)
git add assets/nethical_banner_original.png assets/nethical_logo_original.png

# Commit
git commit -m "Add official Nethical branding assets"

# Push
git push origin copilot/add-branding-assets
```

## Alternative: Use Base64 Encoding

If you have the images but can't access the file system directly, you can provide them as base64 strings:

```bash
# Encode images to base64
base64 banner.png > banner_base64.txt
base64 logo.png > logo_base64.txt

# Then decode and save in the repository
base64 -d banner_base64.txt > assets/nethical_banner_original.png
base64 -d logo_base64.txt > assets/nethical_logo_original.png
```

## Troubleshooting

### Issue: Cannot download from GitHub URL

**Cause:** GitHub user-attachments URLs require authentication or are session-specific.

**Solution:** 
1. Open the image in your browser
2. Right-click and "Save image as..."
3. Manually copy to the assets directory

### Issue: Image dimensions are wrong

**Cause:** Original images might already be at the wrong size.

**Solution:**
The resize script will automatically adjust them. Just ensure you place them as `*_original.png` files.

### Issue: Images look pixelated

**Cause:** Original images might be too small or low quality.

**Solution:**
Ensure the original images are high resolution:
- Banner: At least 1050x300px (preferably higher)
- Logo: At least 256x256px (preferably higher)

## Current Status

✅ Infrastructure is complete and ready
✅ Placeholder images demonstrate functionality
⚠️ Waiting for original branded images to be placed

Once you've placed the original images and run the resize script, the branding will be complete!

## Need Help?

If you're unable to complete this manually, you can:
1. Ask the repository owner to commit the images directly
2. Provide the images through a different hosting service (Dropbox, Google Drive, etc.)
3. Create an issue with the images attached

## Files You Need

1. **Banner (Original)** → `assets/nethical_banner_original.png`
   - Recommended minimum size: 1050x300px or larger
   - Format: PNG, JPG, or any format Pillow can read

2. **Logo (Original)** → `assets/nethical_logo_original.png`
   - Recommended minimum size: 256x256px or larger  
   - Format: PNG with transparency preferred

After placing these files, run:
```bash
python3 scripts/create_branding_assets.py
```

And you're done! ✅
