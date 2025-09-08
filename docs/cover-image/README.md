# Cover Images Directory

This directory contains cover images used by the ContentRunway Publisher Agent for automatic cover image selection.

## Directory Structure

```
docs/cover-image/
├── blog/           # Cover images for Blog category content
├── product/        # Cover images for Product category content
└── README.md       # This file
```

## Usage

The Publisher Agent automatically selects cover images based on:

1. **Category Classification**: Content classified as "Blog" uses images from `blog/` folder, "Product" uses images from `product/` folder
2. **Content Analysis**: Images are selected based on filename matching with content keywords, domain, and title words
3. **Text Removal**: Selected images are processed to remove text automatically

## Supported Image Formats

- PNG (.png)
- JPEG (.jpg, .jpeg)
- BMP (.bmp)
- TIFF (.tiff)

## Image Requirements

- **Recommended Size**: 800x600 or larger for best quality
- **Format**: PNG preferred for transparency support
- **Naming**: Use descriptive names that match your content domains (e.g., `ai-technology.png`, `insurance-digital.jpg`)

## Adding New Images

1. Place images in the appropriate category folder (`blog/` or `product/`)
2. Use descriptive filenames that include relevant keywords
3. Ensure images are high quality and suitable for cover use
4. The system will automatically detect and use new images

## Text Removal

The system automatically processes selected images to remove text overlays while preserving graphics and design elements. This ensures clean, professional cover images for publication.

## Fallback Behavior

If no images are available in the required category folder, the system will:

1. Create a simple placeholder image
2. Log the missing images for dashboard visibility
3. Continue with the publishing process

## Example Filenames

### Blog Category
- `ai-insights-general.png`
- `technology-trends.jpg`
- `digital-transformation.png`
- `data-analysis-blog.jpg`

### Product Category
- `software-platform.png`
- `ai-tool-product.jpg`
- `insurance-solution.png`
- `saas-product.jpg`