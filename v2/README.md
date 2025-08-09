
# docfix-poc v2 (Grid + Coin)

Optimized for photos like a coin on top of a ruled/grid notebook:
- Rectify rotation using dominant grid lines.
- Suppress straight grid lines for circle detection.
- Detect coin circle (center + radius), output overlay and JSON.

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
python main.py your_image.jpg --outdir out
# optional: constrain expected radius range (pixels)
python main.py your_image.jpg --minr 60 --maxr 300
```

Outputs:
- `out/rectified.png`
- `out/overlay.png`
- `out/result.json`

Notes:
- If your image is very high-res, you can downscale with `--scale 0.8` for speed.
- If detection is weak, try adjusting `--minr/--maxr` based on coin size in the rectified image.
