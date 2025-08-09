# DocFix — Document Rectification & Concentric Circle Detection

**DocFix** is a Python CLI tool for:
- Rectifying document photographs so vertical/horizontal lines are truly aligned.
- Detecting concentric circles (e.g., coins, targets) even with grids and perspective distortion.
- Outputting measurement data (center, radius) for analysis or quality control.

This version (**v3.1**) is a production-ready CLI with improved robustness for uneven lighting and low-contrast edges.

---

## Features
- **Grid-based rectification** — aligns documents using dominant horizontal/vertical lines.
- **Lighting correction** — compensates for uneven illumination.
- **Grid suppression** — removes interfering straight lines for better circle detection.
- **Multi-stage detection** — adaptive parameter tuning to handle difficult images.
- **Batch mode** — process a folder of images at once.
- **JSON output** — easy integration with other systems.

---

## Installation
Requires **Python 3.9+**.

```bash
# Inside the v3.1/ folder
pip install -e .
This registers the docfix command globally in your environment.

Usage
Single image
bash
Copy
Edit
docfix path/to/image.jpg --outdir out
Batch mode (folder)
bash
Copy
Edit
docfix path/to/folder --outdir out
With radius constraints
bash
Copy
Edit
docfix image.jpg --outdir out --minr 250 --maxr 450
With intermediate debug images
bash
Copy
Edit
docfix image.jpg --outdir out --save-steps
Output
For each image processed:

rectified.png — aligned document.

overlay.png — detected circle (green) with center (red dot).

result.json — detection results.

Example result.json:

json
Copy
Edit
{
  "center": [667.0, 1031.0],
  "radius": 353.0,
  "confidence": 1.0
}
Examples
Original	Rectified	Overlay

(Replace these with your own captures)

⚙ Parameters
--outdir — output folder (default: out).

--minr / --maxr — expected radius range (pixels).

--scale — scale image before processing (e.g., 0.8 to speed up).

--save-steps — saves debug images (edges, grid suppression) to help tuning.

License
MIT License © 2025 Roberto Carlos Rodriguez

Roadmap
Add tolerance checking (OK / FAIL).

Optional distortion correction (radial).

GUI mode for manual inspection.

Integration with measurement tools.

Contributing
Pull requests are welcome!
For major changes, please open an issue first to discuss what you would like to change.