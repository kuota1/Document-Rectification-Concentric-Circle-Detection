
# docfix-poc

Minimal proof-of-concept to:
1) Rectify a document photo (perspective).
2) (Basic) remove residual skew.
3) Detect concentric circles and measure radii.
4) Emit overlay and JSON.

## Quick start
```bash
pip install -r requirements.txt
python main.py your_image.jpg --outdir out
```
Outputs in `out/`: `rectified.png`, `overlay.png`, `result.json`.

> This is a PoC: robust lattice/affine residual and partial-arc fitting are stubbed
> and will be improved as we iterate on your samples.
```
