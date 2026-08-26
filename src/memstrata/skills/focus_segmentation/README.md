# Focus Segmentation Skill

This skill provides explicit in-focus vs out-of-focus (defocused) plane segmentation from a single image using a reblur gradient ratio algorithm.

## Features

- **Reblur Gradient Ratio**: Analyzes local high-frequency details by comparing gradient energy before and after controlled Gaussian re-blurring.
- **Background Defocus Share**: Calculates the ratio of defocused pixels in background regions to detect shallow depth-of-field (DoF).
- **Defocus Fill**: Replaces defocused-structure pixels with a solid fill color (useful for masking out blurry background elements).
- **Overlay Visualization**: Generates an RGB overlay image highlighting in-focus vs defocused regions.

## Usage

```python
from memstrata.skills.focus_segmentation import focus_map, overlay

# Generate focus map
fmap = focus_map("frame.png")

print(f"In-focus ratio: {fmap.in_focus_ratio:.2%}")

# Check if a specific region (bounding box) is in focus
is_in_focus = fmap.region_in_focus((100, 100, 300, 300)) > 0.5

# Save visualization overlay
overlay("frame.png", fmap, "focus_overlay.png")
```
