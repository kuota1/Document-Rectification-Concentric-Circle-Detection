from __future__ import annotations
import os, json, math, glob
import numpy as np
import cv2 as cv

# ===== pega aquí utilidades de v3: clahe, adaptive_canny, etc. =====
def clahe(gray):
    return cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)

def adaptive_canny(gray):
    v = np.median(gray)
    lo = int(max(0, 0.66*v))
    hi = int(min(255, 1.33*v))
    return cv.Canny(gray, lo, hi, L2gradient=True)

def angle_from_lines(lines):
    if lines is None: return 0.0
    angles = []
    for l in lines:
        x1,y1,x2,y2 = l[0]
        ang = math.atan2(y2-y1, x2-x1)
        if ang < -math.pi/2: ang += math.pi
        if ang >=  math.pi/2: ang -= math.pi
        angles.append(ang)
    if not angles: return 0.0
    return float(np.median(angles))

def rotate_image(img, angle_rad):
    angle_deg = angle_rad * 180/np.pi
    (h,w) = img.shape[:2]
    M = cv.getRotationMatrix2D((w//2,h//2), angle_deg, 1.0)
    return cv.warpAffine(img, M, (w,h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

def rectify_with_grid(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    g = clahe(gray)
    edges = adaptive_canny(g)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=120,
                           minLineLength=min(img.shape[:2])//4, maxLineGap=10)
    ang = angle_from_lines(lines)
    rect = rotate_image(img, -ang)
    return rect, -ang

def suppress_grid_for_circles(gray):
    edges = adaptive_canny(gray)
    k = max(15, min(gray.shape[:2])//40)
    kernel_h = cv.getStructuringElement(cv.MORPH_RECT, (k,1))
    kernel_v = cv.getStructuringElement(cv.MORPH_RECT, (1,k))
    lines_h = cv.morphologyEx(edges, cv.MORPH_OPEN, kernel_h)
    lines_v = cv.morphologyEx(edges, cv.MORPH_OPEN, kernel_v)
    lines = cv.bitwise_or(lines_h, lines_v)
    no_lines = cv.bitwise_and(edges, cv.bitwise_not(lines))
    return no_lines

def detect_circle(gray, minr, maxr):
    blur = cv.GaussianBlur(gray, (5,5), 0)
    edges_wo_grid = suppress_grid_for_circles(gray)
    h, w = gray.shape[:2]
    if minr is None: minr = max(10, int(0.04*min(h,w)))
    if maxr is None: maxr = int(0.45*min(h,w))

    def _hough(img, p1, p2):
        return cv.HoughCircles(img, cv.HOUGH_GRADIENT, dp=1.2,
                               minDist=int(0.8*min(h,w)), param1=p1, param2=p2,
                               minRadius=minr, maxRadius=maxr)

    # intento 1 (normal) y 2 (más sensible)
    circles = _hough(blur, 120, 30)
    if circles is None:
        circles = _hough(edges_wo_grid, 100, 22)
    if circles is None:
        return None

    circles = np.around(circles[0]).astype(np.float32)

    # elegir por gradiente promedio en el perímetro
    gx, gy = np.gradient(blur.astype(np.float32))
    mag = np.hypot(gx, gy)
    scores = []
    for (cx, cy, r) in circles:
        theta = np.linspace(0, 2*np.pi, 360, endpoint=False)
        xs = np.clip((cx + r*np.cos(theta)).astype(int), 0, w-1)
        ys = np.clip((cy + r*np.sin(theta)).astype(int), 0, h-1)
        scores.append(float(np.mean(mag[ys, xs])))
    return tuple(map(float, circles[int(np.argmax(scores))]))

# ===== orquestador =====
def process_image(img, outdir, minr=None, maxr=None, scale=1.0, save_steps=False):
    os.makedirs(outdir, exist_ok=True)
    if scale != 1.0:
        img = cv.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)),
                        interpolation=cv.INTER_AREA)

    rect, _ = rectify_with_grid(img)
    cv.imwrite(os.path.join(outdir, "rectified.png"), rect)

    gray = cv.cvtColor(rect, cv.COLOR_BGR2GRAY)
    circle = detect_circle(gray, minr, maxr)

    result = {"center": None, "radius": None, "confidence": 0.0}
    overlay = rect.copy()
    if circle is not None:
        cx, cy, r = circle
        cv.circle(overlay, (int(cx), int(cy)), int(r), (0,255,0), 2)
        cv.circle(overlay, (int(cx), int(cy)), 3, (0,0,255), -1)
        result.update(center=[float(cx), float(cy)], radius=float(r), confidence=1.0)

    cv.imwrite(os.path.join(outdir, "overlay.png"), overlay)
    with open(os.path.join(outdir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result

def process_path(path, outdir, **kwargs):
    """Acepta un archivo o carpeta. Retorna lista de (in_path, result_json_path)."""
    paths = []
    if os.path.isdir(path):
        exts = (".jpg",".jpeg",".png",".bmp",".tif",".tiff")
        for p in sorted(glob.glob(os.path.join(path, "*"))):
            if p.lower().endswith(exts): paths.append(p)
    else:
        paths = [path]

    outputs = []
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        o = os.path.join(outdir, name)
        os.makedirs(o, exist_ok=True)
        img = cv.imread(p, cv.IMREAD_COLOR)
        if img is None:
            continue
        res = process_image(img, o, **kwargs)
        outputs.append((p, os.path.join(o, "result.json")))
    return outputs
